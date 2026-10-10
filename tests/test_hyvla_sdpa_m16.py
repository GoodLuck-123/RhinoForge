"""HyVLA attention workspace admission and real wrapper bounds, without device."""
from pathlib import Path
import shutil
import subprocess
import pytest
from test_fmb_standalone_layout_rebuild import _definition
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('candidate', (False, True), ids=('m16', 'k128'))
def test_hyvla_m16_attention_admission_and_owner_bounds(tmp_path, candidate):
    cc=shutil.which('c++')
    if cc is None:pytest.skip('C++ compiler unavailable')
    helpers=(ROOT/'src/core/rpu_helpers.h').read_text()
    header=(ROOT/'src/ops/rpu_hyvla_sdpa_tiling.h').read_text()
    sdpa=(ROOT/'src/ops/rpu_sdpa.cpp').read_text()
    wrapper='\n'.join(_definition(sdpa, name) for name in (
        'static void launch_sdpa_hyvla_s51_checked(',
        'void rpu_launch_sdpa_hyvla_s51_m16(', 'void rpu_launch_sdpa_hyvla_s51_k128('))
    source=r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("rejected");}while(0)
#define SDPA_MAX_VLM_SIZE 400
template<class T>T CeilDiv(T x,T y){return (x+y-1)/y;}
namespace c10{enum class DeviceType{CPU,PrivateUse1};}
namespace at{constexpr int kHalf=1;
struct Device{c10::DeviceType kind=c10::DeviceType::PrivateUse1;int index=0;
 c10::DeviceType type()const{return kind;}bool operator==(const Device&v)const{return kind==v.kind&&index==v.index;}};
struct Tensor{bool live=true,contiguous=true;int dtype=1;Device dev{};int nd=7;
 std::array<int64_t,7> shape{1,20,1,8,8,16,16};uint64_t address=0x100000;
 bool defined()const{return live;}bool is_contiguous()const{return contiguous;}
 Device device()const{return dev;}int scalar_type()const{return dtype;}int dim()const{return nd;}
 int64_t size(int n)const{return shape[n];}void* data_ptr()const{return reinterpret_cast<void*>(address);}
 uint64_t nbytes()const{return shape[1]*8*8*16*16*2;}};}
namespace rhino_lkn{uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
''' + _definition(helpers,'struct SdpaTiling {')+';\n'+_definition(helpers,'inline uint32_t GetMhaAttnUnivVlmNeed(')+'\n'+header.replace('#pragma once','').replace('#include "rpu_helpers.h"','')+r'''
struct SpmAllocator{static constexpr uint64_t SPM_USABLE=8387584;};
struct Alloc{bool live=true,overflow=false;bool is_initialized()const{return live;}
 uint32_t addr(int c,int)const{return overflow&&c==7?0xffffff00u:0x10000000u+uint32_t(c)*0x1000000u;}}alloc;
#define SPM_ALLOC alloc
enum class KernelId{HYVLA_EXPERT_SDPA_M16_S51,HYVLA_EXPERT_SDPA_K128_S51};
int launches=0;
void launch_sdpa_spm_unified_impl(const at::Tensor&,const at::Tensor&,int mask,std::optional<double> scale,
 uint32_t,uint32_t,uint32_t,uint32_t,int64_t q,int64_t nq,int64_t nk,int64_t hd,int64_t k,
 int cores,int vtp,int64_t cache_offset,bool narrow,bool delivery,const SdpaTiling* tile,std::optional<KernelId> id){
 assert(mask==4&&!scale&&q==51&&(k==243||k==259||k==275||k==291)&&nq==16&&nk==8&&hd==128&&cores==8&&vtp==8);
 assert(!cache_offset&&!narrow&&!delivery&&tile&&tile->tile_n==128&&tile->tile_n_v16==8);
 assert(id);
 if(*id==KernelId::HYVLA_EXPERT_SDPA_M16_S51)
  assert(tile->tile_m==16&&tile->tile_m_v16==1&&tile->tile_k==256&&tile->tile_k_v16==16);
 else assert(tile->tile_m==64&&tile->tile_m_v16==4&&tile->tile_k==128&&tile->tile_k_v16==8);
 ++launches;}
''' + wrapper + r'''
template<class F>void reject(F f){int old=launches;bool threw=false;try{f();}catch(const std::exception&){threw=true;}assert(threw&&launches==old);}
int main(){
 assert(GetMhaAttnUnivVlmNeed(1,8,16,8,8,8,4)==172);
 assert(GetMhaAttnUnivVlmNeed(4,8,16,8,8,8,4)==280);
 assert(GetMhaAttnUnivVlmNeed(4,8,8,8,8,8,4)==152);
 auto tile=rpu_hyvla_s51_m16_tiling(51,291,262144);assert(tile&&tile->tile_m==16&&tile->tile_n==128&&tile->tile_k==256);
 assert(!rpu_hyvla_s51_m16_tiling(51,291,262143));
 for(int64_t q:{-1,0,1,50,52,64})assert(!rpu_hyvla_s51_m16_tiling(q,291,262144));
 for(int64_t k:{-1,0,240,290,292,304})assert(!rpu_hyvla_s51_m16_tiling(51,k,262144));
 for(int64_t rows:{243,259,275,291})assert(rpu_hyvla_s51_m16_tiling(51,rows,262144));
 at::Tensor k,v;
 auto run=[&](uint32_t q,uint32_t out,uint32_t tmp,uint32_t mask,int64_t bytes=262144){
 rpu_launch_sdpa_hyvla_s51_m16(k,v,q,out,tmp,mask,51,291,bytes);};
 run(256,0x10000,0x20000,0x60000);assert(launches==1);
 for(int64_t rows:{243,259,275,291}){
  k.shape[1]=v.shape[1]=(rows+15)/16;
  rpu_launch_sdpa_hyvla_s51_m16(k,v,256,0x10000,0x20000,0x60000,51,rows,262144);
  --k.shape[1];--v.shape[1];
  reject([&]{rpu_launch_sdpa_hyvla_s51_m16(k,v,256,0x10000,0x20000,0x60000,51,rows,262144);});
 }
 k.shape[1]=v.shape[1]=20;
 reject([&]{run(257,0x10000,0x20000,0x60000);});
 reject([&]{run(256,512,0x20000,0x60000);});
 reject([&]{run(256,0x10000,0x20000,0x20100);});
 reject([&]{run(256,0x10000,0x20000,0x60000,262143);});
 reject([&]{run(256,0x10000,0x7fff00,0x60000);});
 alloc.overflow=true;reject([&]{run(256,0x10000,0x20000,0x60000);});alloc.overflow=false;
 for(int fault=0;fault<10;++fault){auto old=k;
  if(fault==0)k.dtype=2;if(fault==1)k.dev.kind=c10::DeviceType::CPU;
  if(fault==2)k.nd=6;if(fault==3)k.shape[1]=18;if(fault==4)k.shape[3]=4;
  if(fault==5)k.address+=1;if(fault==6)k.address=(UINT64_C(1)<<40)-256;
  if(fault==7)k.contiguous=false;if(fault==8)k.live=false;if(fault==9)k.dev.index=1;
  reject([&]{run(256,0x10000,0x20000,0x60000);});k=old;
 }
 alloc.live=false;reject([&]{run(256,0x10000,0x20000,0x60000);});
}
'''
    if candidate:
        prefix, main = source.split('int main(){', 1)
        main = (main.replace('rpu_hyvla_s51_m16_tiling', 'rpu_hyvla_s51_k128_tiling')
                    .replace('rpu_launch_sdpa_hyvla_s51_m16', 'rpu_launch_sdpa_hyvla_s51_k128')
                    .replace('262144', '32768').replace('262143', '32767')
                    .replace('tile->tile_m==16', 'tile->tile_m==64')
                    .replace('tile->tile_k==256', 'tile->tile_k==128'))
        source = prefix + 'int main(){' + main
    cpp=tmp_path/'test.cpp';cpp.write_text(source);binary=tmp_path/'test'
    subprocess.run([cc,'-std=c++17','-O1',str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
    subprocess.run([str(binary)],check=True,capture_output=True,text=True)
