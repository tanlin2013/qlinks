from glob import glob
from pathlib import Path


def refactor(string: str) -> str:
    return string.replace("./", "").replace("/", ".").replace("\\", ".").replace(".py", "")


dir_path = Path(__file__).relative_to(Path.cwd()).parent
pytest_plugins = [
    refactor(fixture) for fixture in glob(f"{dir_path}/fixtures/*.py") if "__" not in fixture
]
