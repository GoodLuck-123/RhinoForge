"""Execute retained three-camera GU176 composition, CPU mask proof and owner bounds."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


def test_actual_cpu_mask_proof_and_guarded_composition(tmp_path):
    import torch
    from rpu_backend.adapters.hy_vla.preprocess import prefix_masks
    header = (ROOT / 'src/fused/hyvla_vlm_vision_gu176.h').read_text()
    proof = block(header, 'inline bool hyvla_standard_three_camera_mask(')
    launcher = block(header, 'inline void launch_hyvla_vlm_vision_gu176_w8(')
    masks = []
    for cameras, lengths in ((2, (128, 144, 160, 176, 192)), (3, (192, 208, 224, 240))):
        for rows in lengths:
            # The actual preprocessor supplies the independent mask oracle.
            _, _, mask, _, _ = prefix_masks(n_images=cameras, grid=7,
                lang_masks=torch.ones(1, rows-(2+cameras*58), dtype=torch.bool))
            masks.append('{' + str(cameras) + ',{' +
                         ','.join(str(int(v)) for v in (~mask[0]).tolist()) + '}}')
    source = COMMON + r'''
#include <limits>
#include "ops/rpu_linear_tiling.h"
namespace rhino_lkn {uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;
 uint32_t base=0x20000000;bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t off)const{return base+uint32_t(c)*0x1000000u+off;}
} spm;
#define SPM_ALLOC spm
enum class ValuOpType {SILU,MUL};
int calls=0,expected_rows=240;
std::array<uint32_t,3> address{0x20000000,0x20200000,0x20300000};
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,const at::Tensor& w,uint32_t out,
 int64_t m,int64_t n,int64_t k,int part,int cores,uint32_t bias,const at::Tensor& scale){
 assert(in==address[0] && m==176 && n==6144 && k==2048 && part==1 && cores==8 && bias==0);
 assert((calls==0 && out==address[1] && w.address==0x10000000 && scale.address==0x14000000) ||
        (calls==2 && out==address[2] && w.address==0x12000000 && scale.address==0x15000000));++calls;
}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t in,uint32_t out,int64_t n,ValuOpType op,bool gelu,int cores){
 assert(calls++==1 && in==out && in==address[1] && n==176*768 && op==ValuOpType::SILU && !gelu && cores==8);}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t a,uint32_t b,uint32_t out,int64_t n,ValuOpType op,c10::Half value,int cores){
 assert(calls++==3 && a==address[1] && b==address[2] && out==a && n==176*768 && op==ValuOpType::MUL && value==1 && cores==8);}
void rpu_launch_fill_spm_kernel(uint32_t out,int64_t n,c10::Half value,int cores){
 assert(calls++==4 && out==address[1]+176*768*2 && n==(expected_rows-176)*768 && n>0 && value==0 && cores==8);}
PROOF
LAUNCH
int main(){
 std::vector<std::pair<int,std::vector<float>>> masks{MASKS};
 for(const auto& item:masks){const int cameras=item.first;const auto& text=item.second;
  const auto proof=hyvla_standard_three_camera_mask;
  if(cameras==2){assert(!proof(text.data(),text.size()));continue;}
  assert(proof(text.data(),text.size()));
  for(size_t i=0;i<text.size();++i){auto bad=text;bad[i]=1-bad[i];assert(!proof(bad.data(),bad.size()));}
  for(float x:{0.5f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
   auto bad=text;bad[2+cameras*58]=x;assert(!proof(bad.data(),bad.size()));}
 }
 assert(!hyvla_standard_three_camera_mask(nullptr,240));
 for(int rows:{0,128,176,191,200,241,256})assert(!hyvla_standard_three_camera_mask(masks.back().second.data(),rows));
 at::Tensor gw,uw,gs,us;gw.dtype=uw.dtype=at::kChar;gw.shape=uw.shape={6144,2048};uw.address=0x12000000;
 gs.shape=us.shape={6144};gs.address=0x14000000;us.address=0x15000000;
 auto invoke=[&](at::Tensor g,at::Tensor u,at::Tensor s,at::Tensor t,int rows,
                 uint32_t in,uint32_t out,uint32_t up,bool valid){
  calls=0;expected_rows=rows;address={in,out,up};bool accepted=true;
  try{launch_hyvla_vlm_vision_gu176_w8(in,out,up,rows,g,u,s,t);}catch(const std::exception&){accepted=false;}
  assert(accepted==valid && calls==(valid?5:0));};
 for(int rows:{192,208,224,240}){
  invoke(gw,uw,gs,us,rows,0x20000000,0x20200000,0x20300000,true);
  auto effective=rpu_pl_tiling::select_tile_acc16(176,768,2048,false);
  auto full=rpu_pl_tiling::select_tile_acc16(rows,768,2048,false);
  assert(effective.m_tile==400 && effective.n_tile==80 && full.m_tile==effective.m_tile && full.n_tile==effective.n_tile);
  // Exact full-S bound succeeds; storage sized only for effective M must fail.
  for(int region=0;region<3;++region){
   std::array<uint32_t,3> a{0x20000000,0x20200000,0x20300000};
   const uint32_t width=region==0?2048:768;
   a[region]=spm.base+SpmAllocator::SPM_USABLE-rows*width*2;
   invoke(gw,uw,gs,us,rows,a[0],a[1],a[2],true);
   a[region]+=256;invoke(gw,uw,gs,us,rows,a[0],a[1],a[2],false);
   a[region]=spm.base+SpmAllocator::SPM_USABLE-176*width*2;
   invoke(gw,uw,gs,us,rows,a[0],a[1],a[2],false);
  }
 }
 for(int rows:{-16,0,112,127,128,129,144,160,176,191,193,241,256})invoke(gw,uw,gs,us,rows,0x20000000,0x20200000,0x20300000,false);
 const int rows=240;
  for(int ti=0;ti<4;++ti)for(int defect=0;defect<8;++defect){std::array<at::Tensor,4> t{gw,uw,gs,us};auto& bad=t[ti];
   if(defect==0)bad.present=false;if(defect==1)bad.contiguous=false;if(defect==2)bad.dtype=at::kFloat;
   if(defect==3)bad.where.kind=c10::DeviceType::CPU;if(defect==4)--bad.shape[0];if(defect==5)bad.where.index=1;
   if(defect==6)bad.address+=128;if(defect==7)bad.address=(uint64_t(1)<<40)-256;
   invoke(t[0],t[1],t[2],t[3],rows,0x20000000,0x20200000,0x20300000,false);}
  for(int region=0;region<3;++region)for(uint32_t bad:{0x1fffff00u,0x20000001u,0x207fff00u}){
   std::array<uint32_t,3> a{0x20000000,0x20200000,0x20300000};a[region]=bad;
   invoke(gw,uw,gs,us,rows,a[0],a[1],a[2],false);}
  invoke(gw,uw,gs,us,rows,0x20000000,0x20000100,0x20300000,false);
  invoke(gw,uw,gs,us,rows,0x20000000,0x20200000,0x20200100,false);

 spm.ready=false;invoke(gw,uw,gs,us,240,0x20000000,0x20200000,0x20300000,false);spm.ready=true;
 spm.base=0xf8ff0000;invoke(gw,uw,gs,us,240,0xf8ff0000,0xf91f0000,0xf92f0000,false);
}
'''
    compile_run(tmp_path, 'vlm_vision_gu176', source.replace('PROOF', proof)
                .replace('LAUNCH', launcher).replace('MASKS', ','.join(masks)))
