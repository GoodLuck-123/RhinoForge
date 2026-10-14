import hashlib
import importlib.util
from pathlib import Path
import subprocess
import pytest

@pytest.fixture
def factory():
    path = Path(__file__).resolve().parents[2] / 'adapters/rhinovla/factory.py'
    spec = importlib.util.spec_from_file_location('_rhinovla_factory_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def assets(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source / 'model.py').write_text('# pinned source\n')
    subprocess.run(['git', '-C', str(source), 'add', 'model.py'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=', 'commit', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'pinned weights')
    return (source, checkpoint, {'source_sha': revision, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})

def test_profile_checks_source_and_checkpoint(factory, assets):
    (source, checkpoint, identity) = assets
    factory._validate_official_assets(source, checkpoint, **identity)
    checkpoint.write_bytes(b'different weights')
    with pytest.raises(ValueError, match='SHA-256'):
        factory._validate_official_assets(source, checkpoint, **identity)

@pytest.mark.parametrize('filename', ['model.py', 'injected.py'])
def test_dirty_or_untracked_source_rejected(factory, assets, filename):
    (source, checkpoint, identity) = assets
    (source / filename).write_text('# changed source\n')
    with pytest.raises(ValueError, match='clean reviewed source'):
        factory._validate_official_assets(source, checkpoint, **identity)

def test_original_factory_keeps_its_asset_contract(factory, assets):
    (source, checkpoint, identity) = assets
    factory._validate_official_assets(source, checkpoint, **identity)
    with pytest.raises(ValueError, match='clean reviewed source'):
        factory._validate_official_assets(source, checkpoint)
    assert factory.OFFICIAL_SOURCE_SHA != factory.V3_SOURCE_SHA
    assert factory.OFFICIAL_CHECKPOINT_SHA256 != factory.V3_CHECKPOINT_SHA256
