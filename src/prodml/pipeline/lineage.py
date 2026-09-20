"""DVC lineage tracking helper for prodml.

Extracts DVC dataset hashes from dvc.lock, .dvc files, or file contents
to log provenance tags in MLflow tracking runs.
"""

import hashlib
from pathlib import Path

import yaml


def compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file or directory (compatible with DVC hash format)."""
    if path.is_file():
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    elif path.is_dir():
        hasher = hashlib.md5()
        for child in sorted(path.rglob("*")):
            if child.is_file():
                hasher.update(child.relative_to(path).as_posix().encode())
                hasher.update(compute_md5(child).encode())
        return hasher.hexdigest() + ".dir"
    return "file-not-found"


def get_dvc_hash(target_path: str | Path, repo_root: str | Path = ".") -> str:
    """Retrieve the DVC md5 hash for a given data path.

    Checks sources in order:
    1. Direct `<target_path>.dvc` file if it exists.
    2. `dvc.lock` file in repo root for matching outs or deps.
    3. Active MD5 calculation of the target file or directory.

    Args:
        target_path: Relative or absolute path to the dataset or directory.
        repo_root: Root directory of the repository containing dvc.lock.

    Returns:
        String MD5 hash representing the DVC data version.
    """
    root = Path(repo_root).resolve()
    target = Path(target_path)
    rel_target = (
        target.relative_to(root).as_posix()
        if target.is_absolute() and target.is_relative_to(root)
        else target.as_posix()
    )

    # 1. Check for dedicated .dvc file (e.g. data/raw/green_tripdata.parquet.dvc)
    dvc_file = root / f"{rel_target}.dvc"
    if dvc_file.exists():
        try:
            with open(dvc_file, encoding="utf-8") as f:
                dvc_meta = yaml.safe_load(f)
            outs = dvc_meta.get("outs", [])
            if outs and "md5" in outs[0]:
                return str(outs[0]["md5"])
        except (OSError, yaml.YAMLError, KeyError):
            pass

    # 2. Check dvc.lock
    lock_file = root / "dvc.lock"
    if lock_file.exists():
        try:
            with open(lock_file, encoding="utf-8") as f:
                lock_data = yaml.safe_load(f)
            stages = lock_data.get("stages", {})
            for stage_info in stages.values():
                # Check outs
                for out in stage_info.get("outs", []):
                    out_path = out.get("path", "")
                    if (
                        out_path == rel_target or out_path.startswith(f"{rel_target}/")
                    ) and "md5" in out:
                        return str(out["md5"])
                # Check deps
                for dep in stage_info.get("deps", []):
                    dep_path = dep.get("path", "")
                    if (
                        dep_path == rel_target or dep_path.startswith(f"{rel_target}/")
                    ) and "md5" in dep:
                        return str(dep["md5"])
        except (OSError, yaml.YAMLError, KeyError):
            pass

    # 3. Fallback: compute active hash
    resolved = root / rel_target if not target.is_absolute() else target
    if resolved.exists():
        return compute_md5(resolved)

    return "unknown-dvc-hash"
