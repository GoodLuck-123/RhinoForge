"""RTN provenance and zero-minus-one decoder over shared indexed INT4 access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .int4_checkpoint import IndexedInt4Checkpoint, RoutedInt4LayerSource, _json
from .quant_scope import RTN_PROFILE_ID, RTN_RECIPE
from rpu_backend.quant.qwen3_5_moe_rtn import unpack_rtn_matrix


class RtnCheckpoint(IndexedInt4Checkpoint):
    """Published RTN source; manifest identity semantics remain unchanged."""

    profile_id = RTN_PROFILE_ID
    decode_matrix = staticmethod(unpack_rtn_matrix)

    def __init__(self, directory, *, num_layers, num_experts, hidden_size,
                 intermediate_size):
        directory = Path(directory).resolve()
        manifest_path = directory / "derivation_manifest.json"
        manifest = _json(manifest_path)
        if (manifest.get("completed") is not True
                or json.dumps(manifest.get("recipe"), sort_keys=True)
                != json.dumps(RTN_RECIPE, sort_keys=True)):
            raise ValueError("RTN checkpoint requires a completed manifest with the exact recipe")
        if manifest.get("quantized_matrices") != num_layers * num_experts * 3:
            raise ValueError("RTN manifest quantized_matrices does not match the model")
        identity = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        quantizer_sha256 = manifest.get("quantizer_sha256")
        if not isinstance(quantizer_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", quantizer_sha256):
            raise ValueError("RTN manifest requires its source quantizer SHA256")
        self.manifest = manifest
        self.identity = identity
        self.quantizer_sha256 = quantizer_sha256
        super().__init__(directory, num_layers=num_layers, num_experts=num_experts,
                         hidden_size=hidden_size, intermediate_size=intermediate_size)

    def check_shard_record(self, filename, stat):
        record = self.manifest.get("output_files", {}).get(filename, {})
        if (record.get("bytes") != stat.st_size
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
            raise ValueError(
                f"RTN shard is absent from completed manifest or has wrong size: {filename}")

    def check_shard_metadata(self, filename, metadata):
        if not isinstance(metadata, dict) or metadata.get("quantization_algorithm") != "RTN":
            raise ValueError(f"RTN shard lacks its algorithm metadata: {filename}")
