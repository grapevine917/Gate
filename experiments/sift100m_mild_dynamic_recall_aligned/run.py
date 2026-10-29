from pathlib import Path
import subprocess
D=Path(__file__).resolve().parent;python=D.parents[1]/'.venv/bin/python'
for script in ['prepare_validate.py','evaluate_test.py','report.py']:
    print('STAGE',script,flush=True)
    subprocess.run([str(python),str(D/script)],check=True)
