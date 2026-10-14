"""Caller-owned Wall base inputs and real token accounting, without device imports."""
import importlib
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = vars(importlib.import_module("examples.runners.wall_oss.base_engine"))


def observation(cameras=3, side=224):
    mask = torch.arange(26) < 7
    return dict(images=torch.arange(cameras*side*side*3).remainder(256).to(torch.uint8).reshape(cameras,side,side,3),
                instruction='Move the object carefully.', proprioception=torch.linspace(0,1,26),
                state_mask=mask, noise=torch.zeros(1,32,26),
                dof_mask=mask.float().reshape(1,1,26).expand(1,32,26).contiguous())


@pytest.mark.parametrize('cameras', (2, 3))
def test_batch_preserves_every_supplied_image_and_instruction(tmp_path, cameras):
    batch = observation(cameras); path = tmp_path/'batch.pt';torch.save(batch,path)
    images,text,state,mask,noise,dof = ENGINE['load_observation_batch'](path,(224,224),cameras)
    import numpy as np
    assert torch.equal(torch.from_numpy(np.stack([np.asarray(x) for x in images])),batch['images'])
    assert text==batch['instruction'] and torch.equal(state,batch['proprioception'])
    assert torch.equal(mask,batch['state_mask']) and torch.equal(noise,batch['noise'])
    assert torch.equal(dof,batch['dof_mask'])
    with pytest.raises(ValueError, match='images.*shape'):
        ENGINE['load_observation_batch'](path,(224,224),5-cameras)


def test_batch_rejects_shape_precision_nonfinite_and_mask_changes(tmp_path):
    for key,value in [('images',torch.zeros(2,224,224,3,dtype=torch.uint8)),
                      ('images',torch.zeros(3,224,224,3)), ('instruction',''),
                      ('noise',torch.full((1,32,26),float('nan'))),
                      ('state_mask',torch.ones(26,dtype=torch.bool)),
                      ('dof_mask',torch.ones(1,32,26))]:
        batch=observation();batch[key]=value;path=tmp_path/'bad.pt';torch.save(batch,path)
        with pytest.raises(ValueError):ENGINE['load_observation_batch'](path,(224,224))
    batch=observation();batch['unknown']=0;torch.save(batch,path)
    with pytest.raises(ValueError):ENGINE['load_observation_batch'](path,(224,224))


def test_lengths_separate_instruction_scaffold_and_image_expansion():
    tokenizer=SimpleNamespace(encode=lambda text,add_special_tokens: [4,5,6],
                              convert_tokens_to_ids=lambda token: 99)
    ids=torch.tensor([[1,2,99,99,3,99,99,4,99,99,7,8,9]])
    grid=torch.tensor([[1,2,4],[1,2,4],[1,2,4]])
    got=ENGINE['text_lengths'](tokenizer,'complete instruction',ids,grid)
    assert got==dict(instruction_tokens=3,non_image_tokens=7,vision_tokens=6,real_prefix_len=13)
    with pytest.raises(ValueError,match='grid'):
        ENGINE['text_lengths'](tokenizer,'complete instruction',ids,grid*2)


def test_engine_arguments_do_not_need_private_profiles(tmp_path):
    ref=tmp_path/'payload.ref';ref.write_bytes(b'host placeholder')
    for side in (224,256):
        args=ENGINE['arguments'](['--mode','candidate','--precision','fp16',
            '--components','small','--fp16',str(tmp_path),'--main-ref',str(ref),
            '--expansion-ref',str(ref),'--image-size',str(side),str(side),
            '--output-dir',str(tmp_path/f'out-{side}')])
        assert args.cpu_threads==8 and args.main_ref==ref and args.expansion_ref==ref


@pytest.mark.parametrize('cameras', (2, 3))
def test_changed_last_camera_and_exact_golden_roster(tmp_path, cameras):
    import numpy as np
    batch=observation(cameras);path=tmp_path/'batch.pt';torch.save(batch,path)
    images,*_=ENGINE['load_observation_batch'](path,(224,224),cameras)
    before=np.stack([np.asarray(x) for x in images])
    for index in (0,-1):
        changed=ENGINE['changed_camera_images'](images,index)
        actual=np.stack([np.asarray(x) for x in changed])
        expected=before.copy();expected[index]=255-expected[index]
        assert np.array_equal(actual,expected) and len(changed)==cameras
        assert np.array_equal(np.stack([np.asarray(x) for x in images]),before)
    golden=dict(instruction=ENGINE['INSTRUCTION'],horizon=32,num_steps=10,
                num_images=cameras,image_size=(224,224),camera_names=ENGINE['CAMERAS'][:cameras],
                dataset_key='berkeley_autolab_ur5',delta_action=False,state_bins=256)
    ENGINE['validate_golden_metadata'](golden,(224,224),cameras)
    for key,value in [('num_images',5-cameras),('camera_names',list(reversed(golden['camera_names']))),
                      ('image_size',(256,256))]:
        with pytest.raises(ValueError,match='exact base fixture'):
            ENGINE['validate_golden_metadata']({**golden,key:value},(224,224),cameras)


@pytest.mark.parametrize('cameras', (2, 3))
def test_real_engine_routes_camera_roster_and_last_camera_replay(tmp_path, monkeypatch, cameras, instruction_residency=None):
    """Run the actual engine with CPU tensors and a fake public device policy."""
    import importlib.util
    import json
    import numpy as np
    import sys
    from types import ModuleType

    engine = importlib.import_module("examples.runners.wall_oss.base_engine")
    batch=observation(cameras,256);path=tmp_path/'batch.pt';torch.save(batch,path)
    ref=tmp_path/'payload.ref';ref.write_bytes(b'host placeholder')
    args=engine.arguments(['--mode','candidate','--precision','fp16','--components','small',
        '--fp16',str(tmp_path),'--batch',str(path),'--main-ref',str(ref),'--expansion-ref',str(ref),
        '--image-size','256','256','--cameras',str(cameras),'--vision-pair',
        '--warmup','1','--runs','1','--cpu-threads','1','--cpu-thread-sweep','2',
        '--output-dir',str(tmp_path/'out')])
    args.instruction_residency = instruction_residency
    environment_bound = []
    monkeypatch.setattr(engine,'cold_environment',lambda args: environment_bound.append(True) or {})
    monkeypatch.setattr(torch,'rpu',SimpleNamespace(set_caching_allocator=lambda value: None,
        set_ddr_flush=lambda value: None,set_ddr_flush_force=lambda value: None),raising=False)
    threads=[1]
    monkeypatch.setattr(torch,'set_num_threads',lambda value: threads.__setitem__(0,value))
    monkeypatch.setattr(torch,'get_num_threads',lambda: threads[0])
    backend=ModuleType('rpu_backend');backend.__file__=str(ROOT/'python/rpu_backend/__init__.py')
    api=ModuleType('rpu_backend.api')
    runtime=ModuleType('rpu_backend.adapters.wall_oss.runtime')
    transformers=ModuleType('transformers');transformers.__version__='host fixture'
    requested_policy = object()
    if instruction_residency is not None:
        graph = ModuleType('rpu_backend.graph')
        def from_environment(**kwargs):
            assert environment_bound == [True]
            assert kwargs == {'batch_snapshot_mode': instruction_residency}
            return requested_policy
        graph.GraphRuntimePolicy = SimpleNamespace(from_environment=from_environment)
        monkeypatch.setitem(sys.modules, 'rpu_backend.graph', graph)
    expected_names=engine.CAMERAS[:cameras];calls=[];prompts=[];constructed=[]

    def prompt(instruction,names,dataset,delta,state):
        assert list(names)==expected_names
        prompts.append((instruction,list(names)))
        return '|'.join(names)

    runtime._build_flow_prompt=prompt
    for name,module in [('rpu_backend',backend),('rpu_backend.api',api),
                        ('rpu_backend.adapters.wall_oss.runtime',runtime),('transformers',transformers)]:
        monkeypatch.setitem(sys.modules,name,module)

    class Cache:
        phase='READY'
        def __init__(self): self.replays=0
        def size(self): return 1
        def cache_invariant_ok(self): return True
        def snapshot(self):
            return [SimpleNamespace(kernel_count=1,replay_count=self.replays,recapture_count=0,
                batch_snapshot_mode=instruction_residency or 'auto', batch_snapshot_available=False,
                batch_snapshot_effective_contracts=0, batch_snapshot_fallback_reason=
                    'policy_off' if instruction_residency == 'off' else 'untrusted_program:host')]

    def component(method):
        item=SimpleNamespace(_graph_cache=Cache(),_rpu_last_execution_plan={'route':'host'})
        def call(*values,**keywords): item._graph_cache.replays+=1
        setattr(item,method,call)
        return item

    class Policy:
        precision='fp16'
        closed=False
        @classmethod
        def from_checkpoint(cls,*pos,**kwargs):
            assert kwargs['camera_names']==expected_names
            if instruction_residency is None:
                assert 'graph_runtime_policy' not in kwargs
            else:
                assert kwargs['graph_runtime_policy'] is requested_policy
            constructed.append(kwargs)
            return cls()
        def __init__(self):
            vision=component('forward');vision._pending_rev=None;vision._last_forward_plans=[{'images':cameras}]
            llm=component('forward_embeds');action=component('denoise_with_mask');action._precision_mode=0
            ids=torch.tensor([[1,2,*([99]*(2*cameras)),3,4]])
            def process(text,images):
                assert text=='|'.join(expected_names) and len(images)==cameras
                return ids,None,torch.tensor([[1,2,4]]*cameras)
            self._vla=SimpleNamespace(vision=vision,llm=llm,action=action,dataset_key='berkeley_autolab_ur5',
                delta_action=False,_fast_imgproc_ok=True,_pe_norm_fold=False,
                _proprioception_string=lambda state,mask: 'state',_process_inputs=process,
                _processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a,**k: [1,2,3],
                    convert_tokens_to_ids=lambda token: 99)),
                _graph_arena_policy=SimpleNamespace(graph_arena_count=5))
        def to(self,device): assert device=='rpu';return self
        def prepare_graphs(self,*,images,prefix_length_range,**kwargs):
            assert len(images)==cameras
            length=2*cameras+4
            assert prefix_length_range==(length,length)
            self.graph_profile={'execution_profiles':[{'real_min':length,'real_max':length,
                                                       'execution_len':length,'chunk_size':length}]}
        def infer(self,*,images,**kwargs):
            pixels=np.stack([np.asarray(image) for image in images]);calls.append(pixels.copy())
            assert pixels.shape==(cameras,256,256,3)
            self._vla.vision.forward(images);self._vla.llm.forward_embeds(None)
            self._vla.action.denoise_with_mask(None)
            value=sum(int(pixels[i,0,0,0])*(i+1) for i in range(cameras))
            return SimpleNamespace(actions=torch.full((32,26),float(value)),
                extra={'prefix_len':2*cameras+4,'real_prefix_len':2*cameras+4,
                       'action_chunk_size_resolved':32})
        def close(self): Policy.closed=True

    api.WallOssPolicy=Policy
    assert engine.main(args)==0
    assert Policy.closed and len(constructed)==len(prompts)==1
    original=batch['images'].numpy();first=original.copy();first[0]=255-first[0]
    last=original.copy();last[-1]=255-last[-1]
    assert all(any(np.array_equal(call,expected) for expected in (original,first,last)) for call in calls)
    assert any(np.array_equal(call,last) for call in calls)
    report=json.loads((args.output_dir/'report.json').read_text())
    assert report['instruction_residency'] == instruction_residency
    for phase in ('graphs_prepared', 'graphs_final'):
        for component in report[phase].values():
            assert component['instruction_snapshots'] == [{
                'batch_snapshot_mode': instruction_residency or 'auto',
                'batch_snapshot_available': False, 'batch_snapshot_effective_contracts': 0,
                'batch_snapshot_fallback_reason': 'policy_off' if instruction_residency == 'off'
                    else 'untrusted_program:host'}]
    assert report['input']['cameras']==expected_names and report['input']['num_images']==cameras
    assert report['input']['changed_last_camera']==expected_names[-1]
    assert report['input']['vision_tokens']==cameras*2
    assert report['checks']['C_changes_action'] and report['checks']['A_after_C_exact']
    sweep=report['cpu_thread_sweep']['2']
    assert all(sweep['checks'].values()) and all(item['exact'] for item in sweep['vs_primary'].values())
    saved=torch.load(args.output_dir/'outputs.pt',weights_only=True)
    assert torch.equal(saved['A'],saved['A_after_C']) and not torch.equal(saved['A'],saved['C'])


@pytest.mark.parametrize('mode', ('auto', 'off'))
def test_optional_instruction_residency_reaches_cold_policy(tmp_path, monkeypatch, mode):
    test_real_engine_routes_camera_roster_and_last_camera_replay(
        tmp_path, monkeypatch, 2, instruction_residency=mode)
