# Host tools

Run these tools from the source checkout. They use the public model registry and do not require a device.

- `export_examples.py --output /path/to/catalog.json` exports validated templates, profile metadata and source identities. Use `--verify /path/to/catalog.json` to check an existing export.
- `run_zoo_example.py --request /path/to/request.json --config-id CONFIG_ID --output /path/to/run --check-config` validates a selected catalog entry and caller overrides. Without `--check-config`, it verifies the bound checkpoint files and runs the unified example entry. The request uses the `rpu-model-zoo/example-run-bundle/v1` schema and includes the exported `source`, `registry_paths` and selected `configs` rows with their asset bindings.
- `install_hf_env_pth.py` sets all three Hugging Face cache variables at Python startup to `RPU_MODEL_CACHE/.hf` and its `hub` directory. Set `RPU_MODEL_CACHE` before running it; rerun after changing the cache root.
- `link_flat_to_hf_cache.py ORGANIZATION/MODEL LOCAL_DIRECTORY --revision REVISION` links a directory under `RPU_MODEL_CACHE` into the Hugging Face cache. Rerunning updates `refs/main` and replaces existing snapshot links or files.

Catalogs and execution records may contain caller-supplied file locations. Store generated files outside the checkout. The `run_*_optimized.py` and `run_wall_oss*.py` names remain import/CLI compatibility adapters; use `examples/run.py` for new commands.
