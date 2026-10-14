"""Indexed access to the admitted official symmetric GPTQ G128 source.

The identity is a cold metadata fingerprint, not a full weight-payload digest.
Optional download manifests are external provenance, never a loading dependency.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .int4_checkpoint import IndexedInt4Checkpoint, _unique_object
from .quant_scope import GPTQ_PROFILE_ID, validate_gptq_config
from rpu_backend.quant.grouped_uint4 import unpack_gptq_sym8_matrix


class GptqCheckpoint(IndexedInt4Checkpoint):
    """The official routed-only direct-zero-8 profile, with no requantization."""

    profile_id = GPTQ_PROFILE_ID
    decode_matrix = staticmethod(unpack_gptq_sym8_matrix)

    def __init__(self, directory, *, num_layers, num_experts, hidden_size,
                 intermediate_size):
        directory = Path(directory).resolve()
        if (directory / "derivation_manifest.json").exists():
            raise ValueError("GPTQ source must not carry an RTN derivation manifest")
        config_bytes = (directory / "config.json").read_bytes()
        config = json.loads(config_bytes, object_pairs_hook=_unique_object)
        # Presence, including null, is a conflict: never fall through from RTN.
        if "rtn_metadata" in config or "quant_config" in config:
            raise ValueError("GPTQ source conflicts with RTN or mixed-FP8 metadata")
        validate_gptq_config(config)
        super().__init__(directory, num_layers=num_layers, num_experts=num_experts,
                         hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.identity_descriptor = {
            "schema_version": "qwen35_moe_int4_metadata_identity_v1",
            "identity_kind": "cold_metadata_fingerprint_not_payload_hash",
            "profile_id": self.profile_id,
            "encoding": "gptq_sym_direct_zero8_g128",
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": self.index_sha256,
            "shards": self.shard_descriptors,
        }
        self.identity = hashlib.sha256(json.dumps(
            self.identity_descriptor, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()

    def check_shard_record(self, filename, stat):
        # Stock HF directories do not require a producer-specific sidecar.
        pass

    def check_shard_metadata(self, filename, metadata):
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            raise ValueError(f"GPTQ shard has invalid metadata: {filename}")
        if any("rtn" in str(key).lower() or "rtn" in str(value).lower()
               for key, value in metadata.items()):
            raise ValueError(f"GPTQ shard contains conflicting RTN metadata: {filename}")
