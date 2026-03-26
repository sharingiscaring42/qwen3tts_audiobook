"""Pytest configuration: add repo root and client/ to sys.path."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make top-level packages (server/, local/, client/) importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Make client utilities importable as bare modules (e.g. `import utils`)
CLIENT_DIR = REPO_ROOT / "client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))
