import json as _json
import sys

def main() -> int:
    import torch
    import rpu_backend
    for op_name in ('pi05_create', 'pi05_destroy', 'pi05_set_weights', 'pi05_forward'):
        assert hasattr(torch.ops.rpu, op_name), f'torch.ops.rpu.{op_name} not registered - check src/core/rpu_backend.cpp'
    handle = torch.ops.rpu.pi05_create()
    assert isinstance(handle, int), f'pi05_create returned {type(handle)}, expected int'
    assert handle >= 0, f'pi05_create returned negative handle {handle}'
    torch.ops.rpu.pi05_destroy(handle)
    double_destroy_raised = False
    try:
        torch.ops.rpu.pi05_destroy(handle)
    except Exception:
        double_destroy_raised = True
    h_a = torch.ops.rpu.adarms_create()
    torch.ops.rpu.adarms_destroy(h_a)
    adarms_double_destroy_raised = False
    try:
        torch.ops.rpu.adarms_destroy(h_a)
    except Exception:
        adarms_double_destroy_raised = True
    template_parity_ok = double_destroy_raised == adarms_double_destroy_raised
    set_weights_schema = str(torch.ops.rpu.pi05_set_weights.default._schema)
    forward_schema = str(torch.ops.rpu.pi05_forward.default._schema)
    finding_1_set_weights_ok = 'action_in_proj_w' in set_weights_schema and 'action_out_proj_w' in set_weights_schema and ('time_mlp' not in set_weights_schema)
    finding_1_forward_ok = 'adarms_cond_list' in forward_schema and 'suffix_embs_list' not in forward_schema
    finding_8_runtime_num_steps_ok = 'num_steps' in forward_schema and 'num_steps' not in set_weights_schema and ('num_denoise_steps' not in set_weights_schema)
    cr_r5_blocker_2_no_position_ids_ok = 'position_ids' not in forward_schema
    h2 = torch.ops.rpu.pi05_create()
    assert h2 >= 0
    torch.ops.rpu.pi05_destroy(h2)
    result = {'plan': '03-02', 'pi05_create_registered': True, 'pi05_destroy_registered': True, 'pi05_set_weights_registered': True, 'pi05_forward_registered': True, 'create_destroy_roundtrip': True, 'double_destroy_raised': bool(double_destroy_raised), 'adarms_double_destroy_raised': bool(adarms_double_destroy_raised), 'template_parity_ok': bool(template_parity_ok), 'finding_1_set_weights_ok': bool(finding_1_set_weights_ok), 'finding_1_forward_ok': bool(finding_1_forward_ok), 'finding_8_runtime_num_steps_ok': bool(finding_8_runtime_num_steps_ok), 'cr_r5_blocker_2_no_position_ids_ok': bool(cr_r5_blocker_2_no_position_ids_ok)}
    print(f'RESULT: {_json.dumps(result, sort_keys=True)}')
    all_ok = all((v is True for (k, v) in result.items() if k.endswith('_registered') or k == 'create_destroy_roundtrip' or k == 'template_parity_ok' or k.startswith('finding_') or k.startswith('cr_r5_')))
    print(f"[{('PASS' if all_ok else 'FAIL')}] Plan 03-02 pi05 driver smoke")
    return 0 if all_ok else 1
if __name__ == '__main__':
    sys.exit(main())
