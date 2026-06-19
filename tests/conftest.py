from pathlib import Path


def _fixture_module_name(path: Path) -> str:
    relative = path.with_suffix("").relative_to(Path(__file__).parent)
    return ".".join(("tests", *relative.parts))


_FIXTURES_DIR = Path(__file__).with_name("fixtures")
pytest_plugins = [
    _fixture_module_name(path)
    for path in sorted(_FIXTURES_DIR.rglob("*.py"))
    if not path.name.startswith("__")
]
