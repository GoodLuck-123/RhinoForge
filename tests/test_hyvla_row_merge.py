"""Exercise the real row-merge launcher before its graph-aware kernel fetch."""
from test_hyvla_paired_kv import COMMON,ROOT,block,compile_run

def test_actual_launcher_alias_bounds_and_register_geometry(tmp_path):
    launch=block((ROOT/'src/fused/hyvla_row_merge.h').read_text(),'inline void launch_hyvla_row_merge_spm(')
    source=COMMON+r'''
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;
 bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}
} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,68> regs{};void reset_regs(){regs.fill(0);}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t,int)=delete;
} kernel;
int fetched=0,queued=0;bool available=true;
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Kernel* get_kernel_reset(const std::string& s){assert(s=="hyvla_mot_row_merge_s240");
  ++fetched;return available?&kernel:nullptr;}};
struct Queue {bool broadcast=false;int blocks=0;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid[1]==1 && grid[2]==1);
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));blocks=grid[0];++queued;}
} queue;
#define GET_QUEUE(n) (&::queue)
LAUNCH
int main(){
 const uint32_t base=spm.base,txt=base+0x10000,act=base+0x200000,out=base+0x400000;
 auto run=[&](uint32_t t,uint32_t a,uint32_t o,uint32_t mt,uint32_t ma,int c,bool pass,int rows=240,int cores=8){
  int before=fetched,previous=queued;bool ok=true;
  try{launch_hyvla_row_merge_spm(t,a,o,mt,ma,rows,c,cores);}catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==before+int(pass) && queued==previous+int(pass));
 };
 for(int c:{128,256,2048}){
  run(txt,act,out,base,base+512,c,true);
  auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
  assert(pair(4)==base && pair(6)==txt && pair(8)==out && pair(14)==base+512 && pair(16)==act);
  int r=c==2048?1:2;assert(kernel.regs[0]==r && kernel.regs[1]==1);
  assert(kernel.regs[10]==r*32 && kernel.regs[12]==c/16 && kernel.regs[13]==r);
  assert(kernel.regs[18]==2*r && kernel.regs[19]==2*r+r*c/16);
  assert(2*r+2*r*(c/16)<=384 && queue.blocks==(c==2048?15:8));
  run(txt,act,txt,base,base+512,c,true);
  run(txt,act,act,base,base+512,c,false);
  run(txt,txt,out,base,base+512,c,false);
  run(txt,act,txt+256,base,base+512,c,false);
  run(txt,act,act-256,base,base+512,c,false);
  run(txt,act,out,txt,base+512,c,false);
  run(txt,act,out,base,base+256,c,false);
  for(auto bad:{base-256,txt+1,base+SpmAllocator::SPM_USABLE-256}){
   run(bad,act,out,base,base+512,c,false);
   run(txt,bad,out,base,base+512,c,false);
   run(txt,act,bad,base,base+512,c,false);
  }
  run(txt,act,base+SpmAllocator::SPM_USABLE-240*c*2,base,base+512,c,true);
 }
 for(int c:{0,127,512,1024,2049})run(txt,act,out,base,base+512,c,false);
 for(int rows:{128,144,160,176,192,208,224,240})for(int c:{128,256,2048}){
  run(txt,act,out,base,base+512,c,true,rows);
  const int tile=c==2048?16:32,blocks=(rows+tile-1)/tile;
  assert(queue.blocks==blocks && kernel.regs[0]==tile/16);
  assert(kernel.regs[1]==(rows-(blocks-1)*tile)/16);
  run(txt,act,base+SpmAllocator::SPM_USABLE-rows*c*2,base,base+512,c,true,rows);
  run(txt,act,base+SpmAllocator::SPM_USABLE-rows*c*2+256,base,base+512,c,false,rows);
 }
 for(int r:{0,112,127,129,200,239,241,256})run(txt,act,out,base,base+512,128,false,r);
 for(int n:{0,1,4,9})run(txt,act,out,base,base+512,128,false,240,n);
 spm.ready=false;run(txt,act,out,base,base+512,128,false);spm.ready=true;
 spm.base=0xf8ff0000;run(spm.base+0x10000,spm.base+0x200000,spm.base+0x400000,
  spm.base,spm.base+512,2048,false);
}
'''
    compile_run(tmp_path,'row_merge_launcher',source.replace('LAUNCH',launch))
