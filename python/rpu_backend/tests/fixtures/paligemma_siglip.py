from __future__ import annotations

def patch_siglip_from_paligemma_checkpoint(hf_model):
    from rpu_backend.adapters.paligemma import _paligemma_components, _vision_inner
    from rpu_backend.adapters.siglip import patch_siglip_model_for_rpu_all_layers_once
    (_, vision, projector, _, _) = _paligemma_components(hf_model)
    inner = _vision_inner(vision)
    patch_siglip_model_for_rpu_all_layers_once(inner, projector)
    if hasattr(inner, 'encoder'):
        inner.encoder.to('rpu')
    if hasattr(inner, 'post_layernorm'):
        inner.post_layernorm.to('rpu')
    if hasattr(projector, 'to'):
        projector.to('rpu')
    return inner
