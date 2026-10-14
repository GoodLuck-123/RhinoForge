import os
import subprocess
import sys
import torch
import rpu_backend

def run_child_with_env(env_override):
    env = os.environ.copy()
    env.pop('RPU_WARMUP', None)
    env.update(env_override)
    env['_RPU_WARMUP_BINDING_CHILD'] = '1'
    r = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True)
    for line in r.stdout.splitlines():
        if line.startswith('BINDING_VALUE='):
            return int(line.split('=', 1)[1])
    raise RuntimeError(f'Child did not print BINDING_VALUE. stdout:\n{r.stdout}\nstderr:\n{r.stderr}')

def main_parent():
    v = run_child_with_env({})
    assert v == 0, f'Test 1 (unset): expected 0, got {v}'
    print(f'[PASS] Test 1 — RPU_WARMUP unset: {v}')
    v = run_child_with_env({'RPU_WARMUP': '0'})
    assert v == 0, f'Test 2 (N=0): expected 0, got {v}'
    print(f'[PASS] Test 2 — RPU_WARMUP=0: {v}')
    v = run_child_with_env({'RPU_WARMUP': '5'})
    assert v == 5, f'Test 3 (N=5): expected 5, got {v}'
    print(f'[PASS] Test 3 — RPU_WARMUP=5: {v}')
    v = run_child_with_env({'RPU_WARMUP': 'not_an_int'})
    assert v == 0, f'Test 4 (invalid): expected 0, got {v}'
    print(f'[PASS] Test 4 — RPU_WARMUP=not_an_int (graceful fallback): {v}')
    v = run_child_with_env({'RPU_WARMUP': '-3'})
    assert v == 0, f'Test 5 (negative): expected 0, got {v}'
    print(f'[PASS] Test 5 — RPU_WARMUP=-3 (clamp to 0): {v}')
    print('\n[BINDING PASS] torch.rpu.get_rpu_warmup() binding verified.')

def main_child():
    print(f'BINDING_VALUE={torch.rpu.get_rpu_warmup()}')
if __name__ == '__main__':
    if os.environ.get('_RPU_WARMUP_BINDING_CHILD') == '1':
        main_child()
    else:
        main_parent()

def test_warmup_environment_binding():
    main_parent()
