"""Download, checksum-verify, and prepare the public dataset."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.project_data import prepare_dataset


if __name__ == "__main__":
    print(json.dumps(prepare_dataset(), indent=2))