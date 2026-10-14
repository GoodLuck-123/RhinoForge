from unittest.mock import Mock

import pytest
import torch

from rpu_backend.api.hy_embodied import HyEmbodiedPolicy


def _observation(cameras):
    prefix_len = 192 if cameras == 2 else 240
    valid_rows = 2 + cameras * 58 + 1
    mask = torch.zeros(prefix_len, prefix_len, dtype=torch.bool)
    mask[:valid_rows, :valid_rows] = True
    return {
        "images": [torch.zeros(1, 3, 224, 224) for _ in range(cameras)],
        "lang_tokens": torch.zeros(64, dtype=torch.long),
        "prefill_mask": mask,
        "prefill_pos": torch.zeros(prefix_len, dtype=torch.long),
        "modality_mask": torch.zeros(prefix_len, dtype=torch.bool),
        "denoise_mask": torch.zeros(51, prefix_len + 51, dtype=torch.bool),
        "suffix_pos": torch.zeros(51, dtype=torch.long),
        "noise": torch.zeros(1, 50, 32),
        "state": torch.zeros(1, 32),
        "n_valid": valid_rows,
    }


def _policy(tmp_path, cameras):
    policy = HyEmbodiedPolicy.from_checkpoint(tmp_path, n_cameras=cameras)
    policy._rpu_ready = True
    policy._runner = Mock()
    policy._runner.get_action.side_effect = RuntimeError("runner reached")
    return policy


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("location", ["image", "noise", "state", "state_emb"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_observation_is_rejected_before_runner(tmp_path, cameras, location, invalid):
    policy = _policy(tmp_path, cameras)
    obs = _observation(cameras)
    if location == "image":
        tensor = obs["images"][-1]
    elif location == "state_emb":
        del obs["state"]
        tensor = obs["state_emb"] = torch.zeros(1, 1, 1024)
    else:
        tensor = obs[location]
    tensor.reshape(-1)[0] = invalid

    with pytest.raises(ValueError, match="finite values"):
        policy.infer(obs=obs)
    policy._runner.get_action.assert_not_called()


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pose_is_rejected_before_encoding(tmp_path, cameras, invalid):
    policy = _policy(tmp_path, cameras)
    policy._decoder = Mock()
    pose = torch.zeros(16)
    pose[-1] = invalid

    with pytest.raises(ValueError, match="ee_pose.*finite values"):
        policy.infer(images=[torch.zeros(3, 2, 2)] * cameras,
                     instruction="pick", ee_pose=pose)
    policy._decoder.encode_state.assert_not_called()
    policy._runner.get_action.assert_not_called()


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("use_embedding", [False, True])
def test_finite_observation_reaches_runner_unchanged(tmp_path, cameras, use_embedding):
    policy = _policy(tmp_path, cameras)
    obs = _observation(cameras)
    if use_embedding:
        del obs["state"]
        obs["state_emb"] = torch.ones(1, 1, 1024)

    with pytest.raises(RuntimeError, match="runner reached"):
        policy.infer(obs=obs)
    policy._runner.get_action.assert_called_once()
    forwarded = policy._runner.get_action.call_args.kwargs
    assert forwarded.keys() == obs.keys() - {"n_valid"}
    assert all(forwarded[name] is value for name, value in obs.items() if name != "n_valid")
