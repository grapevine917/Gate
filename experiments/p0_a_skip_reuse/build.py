"""Rebuild the frozen benchmark source with the P0-A library/toolchain."""
from pathlib import Path
import json,subprocess
r=Path(__file__).resolve().parent
with (r/'build.log').open('w') as log:
 subprocess.run(json.loads((r/'build.command.json').read_text()),stdout=log,stderr=subprocess.STDOUT,check=True)
