import sys
from pathlib import Path
import pytest

# The tests import the package as ``mlDecoder`` from its parent folder, wherever that is (repo root, MLfolders/, ...).
PARENT = Path(__file__).resolve().parents[2]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


@pytest.fixture(autouse=True)
def isolated_results(tmp_path, monkeypatch):
    """Every test writes datasets and caches under its own temporary results root."""
    monkeypatch.setenv("MLDECODER_RESULTS", str(tmp_path / "results"))
    return tmp_path / "results"
