"""Exercise actual atomic cache publication for G05's generated ACC32 parents."""
from pathlib import Path
import re

from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def test_actual_fixed_nvfp4_acc32_payload_publication_and_withdrawal(tmp_path):
    header = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    source = (ROOT / 'src/core/rpu_kernel_cache.inc').read_text()
    publication = re.search(r'payload_availability_\.publish\(.*?;', source, re.S)[0]
    cpp = r'''
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error("rejected");}while(0)
enum class KernelId:uint16_t __ENUM__;
class KernelPayloadAvailability final __AVAILABILITY__;
struct Cache {
 std::array<void*,static_cast<size_t>(KernelId::_COUNT)> fast_cache_{};
 KernelPayloadAvailability payload_availability_;
 void publish(){KernelPayloadAvailability::ExtendedPayloads pi05_extended_payloads;__PUBLISH__}
};
int main(){
 constexpr KernelId ids[]={KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64,KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48};
 for(int bits=0;bits<4;++bits){Cache cache;
  for(int i=0;i<2;++i)if(bits&(1<<i))cache.fast_cache_[size_t(ids[i])]=reinterpret_cast<void*>(uintptr_t(0x1000+256*i));
  cache.publish();for(int i=0;i<2;++i)assert(cache.payload_availability_.has_loaded(ids[i])==bool(bits&(1<<i)));
  // A new publication withdraws absent parents rather than retaining old bits.
  cache.fast_cache_.fill(nullptr);cache.publish();for(auto id:ids)assert(!cache.payload_availability_.has_loaded(id));
  cache.payload_availability_.withdraw();for(auto id:ids){bool rejected=false;try{cache.payload_availability_.has_loaded(id);}catch(const std::runtime_error&){rejected=true;}assert(rejected);}
 }
 // Existing callers can omit the appended booleans without claiming new payloads.
 KernelPayloadAvailability old;old.publish(true,false);for(auto id:ids)assert(!old.has_loaded(id));
 assert(old.has_loaded(KernelId::RMS_NORM_SPM_V16));
}
'''
    cpp = cpp.replace('__ENUM__', _block(header, 'enum class KernelId'))
    cpp = cpp.replace('__AVAILABILITY__', _block(header, 'class KernelPayloadAvailability final'))
    _compile_run(tmp_path, 'nvfp4_acc32_availability', cpp.replace('__PUBLISH__', publication))
