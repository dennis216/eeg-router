"""Allow lightweight tests without installing the full training dependency set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
