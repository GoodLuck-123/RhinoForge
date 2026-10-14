"""Fetch official mirror weights only when their immutable HF hashes agree."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch_weights(repo_id: str, hf_revision: str, mirror_revision: str, root: Path,
                  *, filenames: tuple[str, ...] | None = None,
                  mirror_repo_id: str | None = None) -> None:
    from huggingface_hub import HfApi

    if (not re.fullmatch(r"[0-9a-f]{40}", hf_revision)
            or not re.fullmatch(r"[0-9a-f]{40}", mirror_revision)):
        raise ValueError("weight transport requires immutable source revisions")
    info = HfApi().model_info(repo_id, revision=hf_revision, files_metadata=True)
    if info.sha != hf_revision:
        raise ValueError("HF metadata did not resolve the requested immutable revision")
    api = f"https://modelscope.cn/api/v1/models/{mirror_repo_id or repo_id}/repo"
    for attempt in range(3):
        try:
            with urlopen(api + "/files?" + urlencode({"Revision": mirror_revision,
                    "Recursive": "true"}), timeout=30) as response:
                rows = json.load(response)["Data"]["Files"]
            break
        except (OSError, HTTPException):
            if attempt == 2:
                raise
    mirror = {row["Path"]: row for row in rows}
    weights = [item for item in info.siblings if (
        item.rfilename in filenames if filenames is not None
        else item.rfilename.endswith(".safetensors"))]
    if not weights:
        raise ValueError("immutable source has no selected weights")
    if filenames is not None and set(filenames) != {item.rfilename for item in weights}:
        raise ValueError("selected weight filename is absent from immutable source")
    for item in weights:
        relative = Path(item.rfilename)
        if (relative.is_absolute() or ".." in relative.parts
                or str(relative) != item.rfilename or "\\" in item.rfilename):
            raise ValueError("weight filename must stay below the task download root")
        if (item.lfs is None or not re.fullmatch(r"[0-9a-f]{64}", item.lfs.sha256)
                or type(item.size) is not int or item.size <= 0):
            raise ValueError(f"invalid immutable HF weight metadata: {item.rfilename}")
        row = mirror.get(item.rfilename, {})
        if (item.lfs is None or row.get("Sha256") != item.lfs.sha256
                or row.get("Size") != item.size):
            raise ValueError(f"official mirror differs from HF source: {item.rfilename}")

    def fetch(item):
        target = root / item.rfilename
        # The content hash in the partial filename prevents an interrupted
        # download from a different revision being resumed into this artifact.
        partial = target.with_name(target.name + f".{item.lfs.sha256}.download")
        if any(not path.resolve().is_relative_to(root.resolve()) for path in
               (target, partial, partial.with_name(partial.name + ".aria2"))):
            raise ValueError("weight destination symlink escapes the task download root")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            url = api + "?" + urlencode({"Revision": mirror_revision,
                                         "FilePath": item.rfilename})
            if shutil.which("aria2c"):
                command = ["aria2c", "--continue=true", "-x16", "-s16", "-k8M",
                    "--file-allocation=none", "--auto-file-renaming=false", "--allow-overwrite=false",
                    "--connect-timeout=30", "--timeout=60", "--max-tries=3",
                    "--summary-interval=60", "--show-console-readout=false", "--console-log-level=warn",
                    "--download-result=hide", "--dir", str(partial.parent), "--out", partial.name, url]
            else:
                command = ["curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "3", "--connect-timeout", "30", "--speed-time", "60",
                    "--speed-limit", "1024", "--continue-at", "-", "--output", str(partial), url]
            subprocess.run(command, check=True)
            candidate = partial
        else:
            candidate = target
        with candidate.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if candidate.stat().st_size != item.size or digest != item.lfs.sha256:
            raise ValueError(f"download differs from immutable HF weight: {candidate}")
        if candidate == partial:
            partial.replace(target)
        print(f"[verified official mirror] {item.rfilename} {digest}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, weights))
