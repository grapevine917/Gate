from pathlib import Path
import json,subprocess
D=Path(__file__).resolve().parent
for dataset,parent in [('sift1m','main_comparison'),('deep1m','deep_p0'),('sift100m','sift100m_p0')]:
 R=D.parent/parent;O=D/dataset;O.mkdir(exist_ok=True);(O/'native').mkdir(exist_ok=True)
 file=R/('build.commands.json' if (R/'build.commands.json').exists() else 'build_commands.json');commands=json.loads(file.read_text());cmd=next(c for c in commands if str(R/'search.cpp') in c)
 cmd=[str(D/'search.cpp') if x==str(R/'search.cpp') else x for x in cmd];cmd[cmd.index('-o')+1]=str(O/'native/search')
 if dataset=='sift100m':cmd.insert(1,'-DQUERY_U8')
 (O/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (O/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('BUILT',dataset,flush=True)
