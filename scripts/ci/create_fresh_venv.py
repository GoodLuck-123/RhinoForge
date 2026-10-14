"""Reuse base dependencies without executing its editable-install startup hooks."""
from pathlib import Path
import sys
import sysconfig
import venv


def create(target: Path, dependency_site: Path | None = None) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"fresh environment already exists: {target}")
    if dependency_site is None:
        dependency_site = Path(sysconfig.get_path(
            "purelib", vars={"base": sys.base_prefix, "platbase": sys.base_prefix}))
    if not dependency_site.is_dir():
        raise FileNotFoundError(dependency_site)
    venv.EnvBuilder(with_pip=True, symlinks=True).create(target)
    site = Path(sysconfig.get_path(
        "purelib", vars={"base": str(target), "platbase": str(target)}))
    # A plain path adds dependencies after this venv's installed packages.
    # Unlike --system-site-packages / addsitedir, it does not run base .pth files.
    (site / "rhinoforge_ci_dependencies.pth").write_text(str(dependency_site.resolve()) + "\n")


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("fresh installation checks require Python 3.12")
    create(Path(sys.argv[1]).absolute())
