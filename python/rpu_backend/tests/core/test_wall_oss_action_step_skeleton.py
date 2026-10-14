import torch
import rpu_backend

def test_ops_registered():
    for op in ('wall_oss_action_step_create', 'wall_oss_action_step_destroy', 'wall_oss_action_step_set_weights', 'wall_oss_action_step_forward'):
        assert hasattr(torch.ops.rpu, op), op

def test_create_destroy():
    h = torch.ops.rpu.wall_oss_action_step_create()
    assert h >= 0
    torch.ops.rpu.wall_oss_action_step_destroy(h)
