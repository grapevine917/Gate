"""Audited loader-only amendment; archive old timed runs before restarting."""
from pathlib import Path
import json,hashlib,shutil,datetime
R=Path(__file__).resolve().parent
proof=json.loads((R/'resident_probe/VERIFIED.json').read_text());assert proof['passed']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(R/'native/search_resident')==proof['search_sha256']
archive=R/'backups/resident_loader/formal_before_amendment';archive.mkdir()
for name in ['p0_a','p0_a_lr','runs','logs','RUN_LOCK.json','evaluation.log','STATUS.json','IO_ACCOUNTING_VERIFIED.json','PROGRESS.json','ALERT.json']:
 p=R/name
 if p.exists():shutil.move(str(p),str(archive/name))
old=json.loads((R/'INPUT_LOCK.json').read_text());assert old==json.loads((R/'backups/resident_loader/INPUT_LOCK.json').read_text())
shutil.copy2(R/'native/search_resident',R/'native/search')
updated=dict(old)
for name in ['native/search','native/online_gate.h']:updated[str(R/name)]=sha(R/name)
changes={k:dict(before=old[k],after=v) for k,v in updated.items() if old[k]!=v};assert len(changes)==2
(R/'INPUT_LOCK.json').write_text(json.dumps(updated,indent=2)+'\n')
info=dict(reason='Loader-only change bounds duplicate file cache during resident q-v loading. Full formal timing rerun; models, data, index, thresholds and config grid unchanged.',changes=changes,old_runs_archive=str(archive),proof=str(R/'resident_probe/VERIFIED.json'),at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(R/'RESIDENT_LOADER_AMENDMENT.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
