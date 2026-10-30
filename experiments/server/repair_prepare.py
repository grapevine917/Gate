#!/usr/bin/env python3
"""Apply only the JSON tuple/list comparison fix to an existing server workspace.

Does not remove/rewrite protocols, configs, models, binaries or measurements.
Validates the old workspace and known predecessor before a backed-up source fix.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

import benchmark as b

PREDECESSOR = '40282ff68430c437780c0b90ab18d39e4dccbf613a7ba1c3fd148fb88296a912'
RELATIVE = b.RERUN+'/prepare.py'


def repair(workspace):
    # The driver itself is unchanged; existing driver/verifier/source/asset
    # hashes must still match. A mismatch is not silently accepted as an update.
    b.check_workspace(workspace)
    new = (b.REPO/RELATIVE).read_text().replace('WSL客户机块层','本次运行环境的Linux块层')
    target = workspace/RELATIVE
    context_path = workspace/'SERVER_CONTEXT.json'
    context = json.loads(context_path.read_text())
    if target.read_text() == new:
        print('prepare.py already has the fix; no changes needed.')
        return
    if b.sha(target) != PREDECESSOR:
        raise ValueError('Unrecognized prepare.py version; refusing to replace it.')
    backup = workspace/'repair_backups'/time.strftime('%Y%m%d_%H%M%S')
    backup.mkdir(parents=True, exist_ok=False)
    (backup/'prepare.py').write_bytes(target.read_bytes())
    (backup/'SERVER_CONTEXT.json').write_bytes(context_path.read_bytes())
    target.write_text(new)
    context['source_hashes'][RELATIVE] = b.sha(target)
    context.setdefault('repairs',[]).append({
        'kind':'prepare_json_tuple_list_comparison', 'before_sha256':PREDECESSOR,
        'after_sha256':b.sha(target), 'backup':str(backup),
        'note':'Only protocol comparison/diagnostics changed; existing protocol and result files retained.'})
    temporary = context_path.with_suffix('.repair.tmp')
    b.dump(temporary,context)
    temporary.replace(context_path)
    b.check_workspace(workspace)
    print('Applied JSON comparison fix. Backup:',backup)
    print('Existing protocols, configurations and native binaries were preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if not (workspace/'SERVER_CONTEXT.json').is_file():
        parser.error('SERVER_CONTEXT.json not found; choose the existing workspace.')
    with (workspace/'server.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        repair(workspace)


if __name__=='__main__':
    main()
