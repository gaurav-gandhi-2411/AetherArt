from pathlib import Path

from setuptools import find_packages, setup


def _parse_requirements(fname: str) -> list[str]:
    """Read install_requires from requirements.txt, skipping torch (needs --index-url)."""
    lines = []
    for line in Path(fname).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        # torch ships from a custom index; omit from install_requires so users
        # can choose CPU or CUDA builds independently.
        if line.lower().startswith("torch"):
            continue
        lines.append(line)
    return lines


setup(
    name="aetherart",
    version="0.1.0",
    packages=find_packages(),
    package_data={"aetherart": ["py.typed"]},
    python_requires=">=3.10",
    install_requires=_parse_requirements("requirements.txt"),
)
