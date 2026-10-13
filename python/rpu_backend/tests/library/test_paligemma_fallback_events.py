import pytest
from rpu_backend.api.errors import RPUBackendError

def test_recorder_records_allowed_event():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    recorder = FallbackRecorder()
    recorder.record(component='paligemma_outer', op='masked_scatter_merge', stage='prefill', reason='first-version CPU merge', shape={'inputs_embeds': [1, 300, 2304]}, dtype='torch.float16', device_from='cpu', device_to='rpu', bytes_moved=300 * 2304 * 2)
    assert recorder.disallowed_count() == 0
    assert recorder.events[0].allowed is True

def test_recorder_marks_unknown_event_disallowed():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    recorder = FallbackRecorder()
    recorder.record(component='projector', op='cpu_projector', stage='prefill', reason='bad fallback', shape={}, dtype='torch.float16', device_from='cpu', device_to='cpu', bytes_moved=0)
    assert recorder.disallowed_count() == 1
    with pytest.raises(RPUBackendError, match='disallowed fallback'):
        recorder.assert_no_disallowed()

def test_aggregate_merges_call_counts():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    recorder = FallbackRecorder()
    for _ in range(2):
        recorder.record(component='lm_head', op='cpu_lm_head', stage='decode', reason='first-version logits fallback', shape={'hidden': [1, 1, 2304]}, dtype='torch.float16', device_from='rpu', device_to='cpu', bytes_moved=2304 * 2)
    rows = recorder.aggregate()
    key = ('lm_head', 'cpu_lm_head', 'decode', 'first-version logits fallback')
    assert rows[key]['call_count'] == 2
    assert rows[key]['bytes_moved'] == 2304 * 2 * 2

def test_negative_bytes_moved_raises():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    recorder = FallbackRecorder()
    with pytest.raises(RPUBackendError, match='bytes_moved'):
        recorder.record(component='lm_head', op='cpu_lm_head', stage='decode', reason='first-version logits fallback', shape={}, dtype='torch.float16', device_from='rpu', device_to='cpu', bytes_moved=-1)

def test_non_positive_call_count_raises():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    recorder = FallbackRecorder()
    with pytest.raises(RPUBackendError, match='call_count'):
        recorder.record(component='lm_head', op='cpu_lm_head', stage='decode', reason='first-version logits fallback', shape={}, dtype='torch.float16', device_from='rpu', device_to='cpu', bytes_moved=0, call_count=0)

def test_bool_counters_raise():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    recorder = FallbackRecorder()
    with pytest.raises(RPUBackendError, match='bytes_moved'):
        recorder.record(component='lm_head', op='cpu_lm_head', stage='decode', reason='first-version logits fallback', shape={}, dtype='torch.float16', device_from='rpu', device_to='cpu', bytes_moved=True)
    with pytest.raises(RPUBackendError, match='call_count'):
        recorder.record(component='lm_head', op='cpu_lm_head', stage='decode', reason='first-version logits fallback', shape={}, dtype='torch.float16', device_from='rpu', device_to='cpu', bytes_moved=0, call_count=True)

def test_shape_is_copied_at_record_time():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    shape = {'x': [1]}
    recorder = FallbackRecorder()
    recorder.record(component='lm_head', op='cpu_lm_head', stage='decode', reason='first-version logits fallback', shape=shape, dtype='torch.float16', device_from='rpu', device_to='cpu', bytes_moved=0)
    shape['x'].append(2)
    assert recorder.events[0].shape == {'x': [1]}

def test_aggregate_uses_explicit_call_count():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    recorder = FallbackRecorder()
    recorder.record(component='lm_head', op='cpu_lm_head', stage='decode', reason='first-version logits fallback', shape={'hidden': [1, 1, 2304]}, dtype='torch.float16', device_from='rpu', device_to='cpu', bytes_moved=2304 * 2, call_count=3)
    rows = recorder.aggregate()
    key = ('lm_head', 'cpu_lm_head', 'decode', 'first-version logits fallback')
    assert rows[key]['call_count'] == 3
    assert rows[key]['bytes_moved'] == 2304 * 2
