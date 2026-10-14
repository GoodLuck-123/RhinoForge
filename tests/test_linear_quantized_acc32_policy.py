"""Explicit ACC32 admission retains precision-family metadata boundaries."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


def test_per_handle_acc32_admits_scaled_families_and_rejects_missing_scales(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    guard = _definition(source, "if (force_acc32) {")
    program = r'''
#include <cassert>
#include <stdexcept>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace at { constexpr int kHalf=1,kChar=2,kByte=3,kFloat=4; }
struct Tensor {
 int type=at::kHalf, elements=1;bool present=true;
 int scalar_type()const{return type;}bool defined()const{return present;}
 int numel()const{return elements;}
};
void validate(Tensor weight,Tensor scale,bool fp8_e4m3_weight=false,
              int nvfp4_tensor_scale_spm_addr=0,bool force_acc32=true){
''' + guard + r'''
}
template<class F>void rejected(F f){bool raised=false;try{f();}catch(const std::runtime_error&){raised=true;}assert(raised);}
int main(){
 const Tensor none{at::kHalf,0,false},empty{at::kHalf,0,true};
 const Tensor half{at::kHalf},w8{at::kChar},w4{at::kByte},fp32{at::kFloat};
 validate(half,none);validate(half,empty);
 validate(w8,half); validate(w4,half);
 rejected([&]{validate(half,half);});
 rejected([&]{validate(w8,none);});rejected([&]{validate(w4,none);});
 validate(w4,half,true); // E4M3 family validates its exact layout below
 validate(w4,w4,false,256); // NVFP4 carries its own tensor-scale operand
 rejected([&]{validate(fp32,none);});
 // An ACC16 caller is unchanged; its existing family guard handles metadata.
 validate(w4,w4,false,256,false);
}
'''
    cpp, binary = tmp_path / "guard.cpp", tmp_path / "guard"
    cpp.write_text(program)
    built = subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
