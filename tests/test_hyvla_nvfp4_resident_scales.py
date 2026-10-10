"""Physical scale-byte packing and actual SPM-only launch admission."""
import importlib.util
import re
from pathlib import Path
import pytest
import torch
from test_hyvla_paired_kv import COMMON,ROOT,block,compile_run


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


packing=load('hyvla_resident_scales',ROOT/'python/rpu_backend/quant/hyvla_nvfp4_resident_scales.py')
nv4=load('hyvla_scale_nv4',ROOT/'python/rpu_backend/quant/nvfp4_pack.py')


@pytest.mark.parametrize("layers",[4,8])
def test_scale_bundle_roundtrip_and_logical_k16_core_lane_addresses(layers):
    logical=[]
    for projection in range(2):
        family=[]
        for layer in range(layers):
            rows=torch.arange(64,dtype=torch.int64)[:,None]
            cols=torch.arange(2048,dtype=torch.int64)[None,:]
            family.append(((rows*29+cols*7+(cols//256)*43+layer*31+projection*97)%256).byte())
        logical.append(family)
    installed=[[nv4.swizzle_nvfp4_scales(scale,1,8) for scale in family] for family in logical]
    saved=[[scale.clone() for scale in family] for family in installed]
    carrier=packing.pack_hyvla_nvfp4_resident_scales(*installed)
    assert carrier.shape==(8,layers*16384) and carrier.dtype==torch.float16 and carrier.is_contiguous()
    # Check addresses against logical K16 scales, independently of the inverse.
    actual=carrier.view(torch.uint8).reshape(8,layers,2,32,16,2,16)
    for core in range(8):
        for layer in range(layers):
            for projection in range(2):
                expected=logical[projection][layer][:,core*256:(core+1)*256]
                expected=expected.reshape(32,2,16,16).permute(0,2,1,3)
                assert torch.equal(actual[core,layer,projection],expected)
    restored=packing.unpack_hyvla_nvfp4_resident_scales(carrier)
    for family,snapshot,decoded in zip(installed,saved,restored):
        for actual,expected,roundtrip in zip(family,snapshot,decoded):
            assert torch.equal(actual,expected) and torch.equal(roundtrip,expected)
    assert torch.equal(packing.pack_hyvla_nvfp4_resident_scales(*restored).view(torch.uint8),carrier.view(torch.uint8))


def test_scale_bundle_rejects_wrong_layer_count_type_shape_device_and_noncontiguous():
    good=torch.zeros((64,2048),dtype=torch.uint8)
    family=[good]*8
    for count in (0,1,3,4,5,7,9,24):
        with pytest.raises(ValueError):packing.pack_hyvla_nvfp4_resident_scales([good]*count,family)
        with pytest.raises(ValueError):packing.pack_hyvla_nvfp4_resident_scales(family,[good]*count)
    bad=[good.half(),good[:63],torch.empty((64,2048),device='meta',dtype=torch.uint8),
         torch.empty((64,4096),dtype=torch.uint8)[:,::2],None]
    for value in bad:
        for projection in range(2):
            families=[family.copy(),family.copy()]
            families[projection][7]=value
            with pytest.raises(ValueError):packing.pack_hyvla_nvfp4_resident_scales(*families)
    carrier=packing.pack_hyvla_nvfp4_resident_scales(family,family)
    for value in (carrier.float(),carrier[:7],carrier[:,:65535].contiguous(),carrier[:,:65537].contiguous(),
                  torch.empty((8,131072),device='meta',dtype=torch.float16),
                  torch.empty((8,262144),dtype=torch.float16)[:,::2],None):
        with pytest.raises(ValueError):packing.unpack_hyvla_nvfp4_resident_scales(value)


@pytest.mark.parametrize('swiglu',[False,True])
@pytest.mark.parametrize('scale_layers',[4,8])
def test_actual_spm_scale_launcher_nine_owners_and_source_encoding(tmp_path,swiglu,scale_layers):
    launch=block((ROOT/'src/fused/hyvla_nvfp4_resident_scales.h').read_text(),
                 'inline void launch_hyvla_nvfp4_resident_scales_spm(')
    source=COMMON+r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId {HYVLA_NVFP4_RESIDENT_SCALES_S51,HYVLA_NVFP4_RESIDENT_SCALES_SWIGLU_S51};
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int c,uint32_t o)const{return base+c*0x1000000u+o;}} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,130> regs{};int counters=0;
 void reset_regs(){regs.fill(0);counters=0;}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t i,uint32_t v){assert(i==128 && v==0);regs.at(i)=v;++counters;}
 void set_regs(uint32_t,int)=delete;} kernel;
bool available=true;int fetched=0,queued=0;
Kernel* get_kernel(KernelId id){assert(id==(SWIGLU?KernelId::HYVLA_NVFP4_RESIDENT_SCALES_SWIGLU_S51:KernelId::HYVLA_NVFP4_RESIDENT_SCALES_S51));++fetched;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct Queue {void set_broadcast_mode(bool b){assert(b);}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(grid==std::vector<uint16_t>({8,1,1}) && cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}} queue;
#define GET_QUEUE(n) (&::queue)
// Deliberately no DDR graph/types/flush stubs: this helper must be SPM-only.
LAUNCH
int main(){
 const uint32_t base=spm.base;
 std::array<uint32_t,9>a{base+0x10000,base+0x100000,base+0x110000,base+0x200000,base+0x220000,
  base+0x300000,base+0x304000,base+0x700000,base+0x700100};
 const std::array<uint32_t,9>bytes{51*1024*2,51*256*2,51*256*2,131072,131072,16384,16384,128,128};
 auto run=[&](bool pass,int rows=51,int cores=8,int layer=0,int count=SCALE_LAYERS){const int f=fetched,q=queued;bool ok=true;
  try{launch_hyvla_nvfp4_resident_scales_spm(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],uint16_t(layer),SWIGLU,rows,cores,count);}
  catch(const std::exception&){ok=false;}assert(ok==pass && fetched==f+int(pass) && queued==q+int(pass));};
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 for(int layer:{0,1,SCALE_LAYERS-1}){run(true,51,8,layer);
  assert(pair(0)==a[0] && pair(2)==a[3] && pair(4)==a[1] && pair(32)==a[4] && pair(38)==a[2]);
  assert(pair(20)==a[5] && pair(34)==a[6] && pair(24)==a[7] && pair(36)==a[8]);
  assert(pair(8)==51 && pair(10)==256 && kernel.regs[12]==1024 && kernel.regs[13]==16);
  assert(pair(16)==2048 && pair(18)==512 && pair(22)==512 && kernel.regs[26]==1 && kernel.regs[27]==layer);
  assert(kernel.counters==int(SWIGLU) && kernel.regs[64]==8 && kernel.regs[65]==1 && kernel.regs[66]==1);
  if(SWIGLU){assert(kernel.regs[48]==32512 && kernel.regs[49]==13056 && pair(52)==a[1] && pair(54)==a[2] && pair(56)==a[1]);}
 }
 for(int layer:{SCALE_LAYERS,SCALE_LAYERS+1,23,31,32,65535})run(false,51,8,layer);
 for(int count:{0,1,3,5,7,9,24})run(false,51,8,0,count);
 for(int rows:{0,50,52,64})run(false,rows);
 for(int cores:{0,1,4,9})run(false,51,cores);
 for(size_t i=0;i<a.size();++i){const auto good=a[i];
  for(uint32_t bad:{base-256,good+1}){a[i]=bad;run(false);}a[i]=good;
  for(size_t j=0;j<a.size();++j)if(i!=j){a[i]=a[j];run(false);a[i]=good;
   if(bytes[j]>256){a[i]=a[j]+256;run(false);a[i]=good;}}
  a[i]=base+((SpmAllocator::SPM_USABLE-bytes[i])&~255u);run(true);a[i]+=256;run(false);a[i]=good;
 }
 {available=false;const int f=fetched,q=queued;bool rejected=false;
  try{launch_hyvla_nvfp4_resident_scales_spm(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],0,SWIGLU);}
  catch(const std::exception&){rejected=true;}assert(rejected && fetched==f+1 && queued==q);available=true;}
 spm.ready=false;run(false);spm.ready=true;
 spm.base=0xf8ff0000;for(auto& p:a)p+=spm.base-base;run(false);
}
'''
    source=re.sub(r'\bSWIGLU\b','true' if swiglu else 'false',source.replace('LAUNCH',launch))
    source=source.replace('SCALE_LAYERS',str(scale_layers))
    compile_run(tmp_path,'nvfp4_resident_scales',source)
