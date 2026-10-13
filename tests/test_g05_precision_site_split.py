"""Precision variants retain distinct source-authority sites in actual model helpers."""
from pathlib import Path
import re

from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def test_actual_mode_sites_are_distinct_and_preserve_original_routes(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    names = ('text_gateup_pair_site', 'action_gateup_pair_site',
             'action_nvfp4_swiglu_site', 'action_nvfp4_o_resident_site',
             'action_nvfp4_down_resident_site')
    methods = '\n'.join('int64_t '+n+'()const'+_block(source, 'int64_t Qwen3_5Model::'+n+'(') for n in names)
    ids = set(re.findall(r'QWEN35_[A-Z0-9_]+', methods))
    constants = '\n'.join(re.search(r'constexpr int64_t '+n+r'\s*=.*?;', source, re.S)[0] for n in sorted(ids))
    cpp = r'''
#include <cassert>
#include <cstdint>
__CONSTANTS__
struct Model{
 bool linear_acc32_=true,text_gateup_pair_w8_=false,action_gateup_pair_w8_=false,action_gateup_pair_nvfp4_=false;
 __METHODS__
};
int main(){
 Model m;
 assert(m.text_gateup_pair_site()==751944759910871833LL);
 assert(m.action_gateup_pair_site()==5257909631264440265LL);
 m.linear_acc32_=false;
 assert(m.text_gateup_pair_site()==2576448788894560734LL);
 assert(m.action_gateup_pair_site()==4279327644766908377LL);
 m.text_gateup_pair_w8_=m.action_gateup_pair_w8_=true;
 for(bool mode:{false,true}){m.linear_acc32_=mode;assert(m.text_gateup_pair_site()==267678052960427182LL&&m.action_gateup_pair_site()==4700085927040894734LL);}
 m.action_gateup_pair_w8_=false;m.action_gateup_pair_nvfp4_=true;
 m.linear_acc32_=false;
 assert(m.action_gateup_pair_site()==4571887477213075828LL);
 assert(m.action_nvfp4_swiglu_site()==1871355692309183287LL);
 assert(m.action_nvfp4_o_resident_site()==5103834779006160382LL);
 assert(m.action_nvfp4_down_resident_site()==5936546264671002449LL);
 m.linear_acc32_=true;
 assert(m.action_gateup_pair_site()==9073505328675940699LL);
 assert(m.action_nvfp4_swiglu_site()==4752878381648738023LL);
 assert(m.action_nvfp4_o_resident_site()==4187530286635807349LL);
 assert(m.action_nvfp4_down_resident_site()==8057254336740368708LL);
}
'''
    cpp = cpp.replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>')
    _compile_run(tmp_path, 'actual_precision_sites', cpp.replace('__CONSTANTS__', constants).replace('__METHODS__', methods))
