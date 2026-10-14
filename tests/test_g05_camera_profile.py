"""Cold source selection and two-camera host ownership; no native extension."""
import ast
import copy
import functools
from collections.abc import Mapping
from pathlib import Path
import threading
import types
import weakref

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def runtime_helpers():
    """Execute the actual small CPU call path without loading an device module."""
    source = ROOT / "python/rpu_backend/adapters/g05/runtime.py"
    names = {
        "_snapshot_instance_attrs", "_restore_instance_attrs", "_instance_method_is",
        "_is_vla_rtc_fm_release", "_g05_release_image_specs", "_g05_camera_profile_matches",
        "configure_g05_camera_profile", "_rpu_g05_release_image_transform",
        "_g05_release_processor_runtime_ready", "_rpu_g05_release_preprocess",
        "patch_g05_release_processor_for_rpu", "_rpu_g05_to_tensor",
        "configure_g05_vision_camera_batch", "_rpu_g05_forward_embed",
    }
    constants = {"_G05_RELEASE_CAMERA_NAMES", "_G05_TWO_CAMERA_NAMES", "_G05_RELEASE_IMAGE_SPECS"}
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id in constants for x in node.targets):
            nodes.append(node)
    namespace = dict(torch=torch, copy=copy, functools=functools, Mapping=Mapping,
                     types=types, weakref=weakref, _POLICY_INSTALL_LOCK=threading.Lock())
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def named_type(name, module, **methods):
    return type(name, (), {"__module__": module, **methods})


def source_fixture():
    cfg = types.SimpleNamespace(num_input_images=3, cond_steps=1, num_obs_steps=1,
        position_ids_type="pi0fast", ae_vlm_condition_mode="cross_attn_only")
    vision_cfg = types.SimpleNamespace(temporal_freq=0, spacetime_mode="factorized",
        token_drop_layer=None, batch_all_cameras=False, patch_size=16,
        temporal_patch_size=2, spatial_merge_size=2, in_channels=3,
        out_hidden_size=2048, hidden_act="gelu_pytorch_tanh")
    fm = types.SimpleNamespace(horizon_steps=32, action_dim=27, num_inference_steps=10,
                              time_convention="pi_convention", action_causal=False)
    model = named_type("G05ModelQwen35", "vla_rtc.models.g05.g05_model_qwen35")()
    model.cfg, model.vision_tower, model.fm_helper = cfg, types.SimpleNamespace(config=vision_cfg), fm
    model.vlm, model.action_expert, model._cached_image_grid_thw = types.SimpleNamespace(), types.SimpleNamespace(), None
    policy = named_type("G05PolicyQwen35", "vla_rtc.models.g05.g05_policy_qwen35", parameters=lambda self: [self.weight])()
    policy.model, policy.weight, policy.training = model, torch.zeros(1), False
    policy.num_input_images, policy.config, policy.model_config = 3, cfg, cfg
    policy.processor = types.SimpleNamespace(model_cfg=cfg)
    policy.continuous_action = policy.discrete_action = policy.return_continuous_action = True
    policy.predict_cot, model.use_training_rtc, model.max_inference_delay = False, True, 8
    names = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
    tokenizer = named_type("G05BenchTokenizer", "vla_rtc.models.g05.bench_adapter")()
    tokenizer.num_input_images, tokenizer.image_sizes, tokenizer.cot_mode = 3, {name: (256, 256) for name in names}, "none"
    def preprocess(self, data):
        result = {}
        for item in self.shape_meta["images"]:
            image = data["images"][item["key"]]
            for transform in self.val_transforms[item["key"]]:
                image = transform(image)
            result[item["key"]] = image
        return result
    processor = named_type("BaseProcessor", "vla_rtc.processors.base_processor", preprocess=preprocess)()
    processor.is_train, processor.num_obs_steps, processor.num_output_cameras = False, 1, 3
    processor.action_output_dim = processor.proprio_output_dim = 27
    processor.pixel_values_format, processor.tokenizer = "dict", tokenizer
    raw = ((3,360,640),(3,480,640),(3,480,640))
    processor.shape_meta = {"images": [dict(key=n,raw_shape=r,shape=(3,256,256)) for n,r in zip(names,raw)], "state": [{"key":"state","shape":27}]}
    def resize(self, image):
        return torch.nn.functional.interpolate(image.float(), size=self.size, mode="nearest").to(torch.uint8)
    Resize=named_type("Resize", "torchvision.transforms.transforms", __call__=resize)
    def to_tensor(self, image):
        return image.to(torch.float32)/255.
    ToTensor=named_type("ToTensor", "vla_rtc.transforms.image", __call__=to_tensor)
    def normalize(self, image):
        return image.sub_(.5).div_(.5) if self.inplace else (image-.5)/.5
    Normalize=named_type("Normalize", "torchvision.transforms.transforms", __call__=normalize)
    def chain():
        r=Resize();r.size=(256,256);r.interpolation="nearest";r.max_size=None;r.antialias=False
        n=Normalize();n.mean=n.std=(.5,.5,.5);n.inplace=False
        return [r,ToTensor(),n]
    processor.val_transforms={name:chain() for name in names}
    processor.train_transforms={name:chain() for name in names}
    return policy, processor


def test_default_three_preserves_original_metadata_and_two_is_explicit():
    runtime = runtime_helpers()
    policy, processor = source_fixture()
    old_meta, old_sizes, old_transforms = processor.shape_meta, processor.tokenizer.image_sizes, processor.val_transforms
    assert runtime.configure_g05_camera_profile(policy,processor) is policy
    assert (processor.shape_meta is old_meta and processor.tokenizer.image_sizes is old_sizes
            and processor.val_transforms is old_transforms)
    assert not hasattr(policy,"_rpu_g05_camera_names")
    assert runtime._is_vla_rtc_fm_release(policy,policy.model)
    policy.num_input_images=2
    assert not runtime._is_vla_rtc_fm_release(policy,policy.model)


def test_two_binds_every_owner_and_rejects_later_metadata_drift():
    runtime=runtime_helpers();policy,processor=source_fixture()
    original_meta=processor.shape_meta
    names=("head_rgb","left_wrist_rgb")
    assert runtime.configure_g05_camera_profile(policy,processor,camera_names=names) is policy
    assert policy.num_input_images==processor.num_output_cameras==processor.tokenizer.num_input_images==2
    assert tuple(processor.tokenizer.image_sizes)==tuple(processor.val_transforms)==tuple(processor.train_transforms)==names
    assert tuple(item["key"]for item in processor.shape_meta["images"])==names
    assert len(original_meta["images"])==3
    assert policy.config.num_input_images==policy.model.cfg.num_input_images==2
    assert runtime._is_vla_rtc_fm_release(policy,policy.model)
    assert runtime.configure_g05_camera_profile(policy,processor,camera_names=names) is policy
    processor.tokenizer.image_sizes["right_wrist_rgb"]=(256,256)
    assert not runtime._is_vla_rtc_fm_release(policy,policy.model)
    with pytest.raises(RuntimeError,match="incomplete source state"):
        runtime.configure_g05_camera_profile(policy,processor,camera_names=names)


@pytest.mark.parametrize("names",[("left_wrist_rgb","head_rgb"),("head_rgb","right_wrist_rgb"),("head_rgb",),(),("head_rgb","left_wrist_rgb","left_wrist_rgb")])
def test_foreign_camera_geometry_rejects_without_mutation(names):
    runtime=runtime_helpers();policy,processor=source_fixture()
    old_meta=processor.shape_meta
    with pytest.raises(ValueError,match="cameras must"):
        runtime.configure_g05_camera_profile(policy,processor,camera_names=names)
    assert processor.shape_meta is old_meta and policy.num_input_images==3


@pytest.mark.parametrize("changed",["policy_installed","vision_installed","session","processor_patched","cpu_used"])
def test_camera_selection_rejects_after_any_runtime_or_cpu_inference(changed):
    runtime=runtime_helpers();policy,processor=source_fixture()
    if changed=="policy_installed":policy._rpu_g05_policy_install_started=True
    if changed=="vision_installed":policy.model.vision_tower._rpu_vision_handle=17
    if changed=="session":policy._execution_session=object()
    if changed=="processor_patched":processor._rpu_g05_release_processor_ready=True
    if changed=="cpu_used":policy.model._cached_image_grid_thw=torch.ones(3,3)
    with pytest.raises(RuntimeError,match="fresh cold"):
        runtime.configure_g05_camera_profile(policy,processor,camera_names=("head_rgb","left_wrist_rgb"))
    assert policy.num_input_images==3 and processor.num_output_cameras==3


def test_partial_configuration_failure_restores_source_owners():
    runtime=runtime_helpers();policy,processor=source_fixture()
    class RejectConfig:
        def __init__(self):self.num_input_images=3;self.once=True
        def __setattr__(self,name,value):
            if name=="num_input_images" and value==2 and getattr(self,"once",False):
                self.once=False
                raise RuntimeError("simulated read-only configuration")
            object.__setattr__(self,name,value)
    policy.processor.model_cfg=RejectConfig()
    original_meta=processor.shape_meta
    with pytest.raises(RuntimeError,match="read-only"):
        runtime.configure_g05_camera_profile(policy,processor,camera_names=("head_rgb","left_wrist_rgb"))
    assert policy.config.num_input_images==policy.model.cfg.num_input_images==policy.processor.model_cfg.num_input_images==3
    assert policy.num_input_images==processor.num_output_cameras==processor.tokenizer.num_input_images==3
    assert processor.shape_meta is original_meta and not hasattr(policy,"_rpu_g05_camera_names")


@pytest.mark.parametrize("count",[2,3])
def test_selected_host_preprocessing_preserves_camera_bytes_and_shared_views(count):
    runtime=runtime_helpers();policy,processor=source_fixture()
    names=runtime._G05_RELEASE_CAMERA_NAMES[:count]
    runtime.configure_g05_camera_profile(policy,processor,camera_names=names)
    generator=torch.Generator().manual_seed(13)
    data={"images":{key:torch.randint(0,256,(1,*raw),dtype=torch.uint8,generator=generator)
                    for key,raw in runtime._G05_RELEASE_IMAGE_SPECS[:count]}}
    original=processor.preprocess(copy.deepcopy(data))
    assert runtime.patch_g05_release_processor_for_rpu(processor) is processor
    output=processor.preprocess(data)
    assert tuple(output)==names and runtime._g05_release_processor_runtime_ready(processor)
    for key in names:assert torch.equal(original[key],output[key])
    assert all(output[key].untyped_storage().data_ptr()==output[names[0]].untyped_storage().data_ptr()for key in names)
    assert runtime.patch_g05_release_processor_for_rpu(processor) is processor
    processor.num_output_cameras=1
    assert not runtime._g05_release_processor_runtime_ready(processor)


def test_two_rejects_b3_and_changed_live_input_before_effects():
    runtime=runtime_helpers();policy,processor=source_fixture()
    runtime.configure_g05_camera_profile(policy,processor,camera_names=("head_rgb","left_wrist_rgb"))
    policy._rpu_g05_fm_only_customer_release=True
    policy._execution_session=types.SimpleNamespace(require_cold=lambda:None)
    with pytest.raises(ValueError,match="selected source camera count"):
        runtime.configure_g05_vision_camera_batch(policy,3)
    with pytest.raises(ValueError,match="installed cold profile"):
        runtime._rpu_g05_forward_embed(policy.model,torch.zeros(1,2,dtype=torch.long),
                                      torch.ones(1,2,dtype=torch.long),{"head_rgb":torch.zeros(1)})
