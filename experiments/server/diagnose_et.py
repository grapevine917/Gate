#!/usr/bin/env python3
"""Replay two original batches with ET traces, without changing frozen files.

Diagnostic timings are not benchmark results. Instrumentation can itself affect
numerical behaviour; compare replay counts with the original CSVs as well.
"""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

KEY = "Expand_9737fe6e75f647ea59d2"
QUERY = 329


def instrument(source):
    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("Unrecognized runner source: " + old[:70])
        source = source.replace(old, new)
    replace_once('  auto search=[&]',
                   '  diskann::TerminationTrace debug_trace;\n  auto search=[&]')
    replace_once('   index.cached_beam_search(',
        '   if(i==329){debug_trace.boundaries.clear();debug_trace.scored_nodes.clear();diskann::termination_trace=&debug_trace;}\n'
        '   index.cached_beam_search(')
    replace_once('   diskann::online_gate=nullptr;',
        '   diskann::termination_trace=nullptr;\n   diskann::online_gate=nullptr;')
    replace_once('  csv<<"query,hits,',
        r'''  if(termination){
   if(termination->uses_vector_features() || termination->query_feature_width()!=0)throw std::runtime_error("Expected state-only ET");
   std::ofstream trace(p+".et.csv");
   trace<<"boundary,node,completed,threshold,score,recomputed";
   for(int j=0;j<17;++j)trace<<",state"<<j;
   trace<<"\n"<<std::setprecision(std::numeric_limits<float>::max_digits10);
   size_t boundary=0;
   for(const auto& v:debug_trace.boundaries){
    trace<<boundary++<<','<<v.node<<','<<v.completed<<','<<termination->threshold<<','<<v.score<<','<<termination->score(nullptr,v.node,v.state.data());
    for(float value:v.state)trace<<','<<value;
    trace<<"\n";
   }
  }
  csv<<"query,hits,''')
    return source


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def compare(left, right):
    report = dict(boundaries=[len(left), len(right)])
    report['recomputed_mismatches'] = [
        sum(r['score'] != r['recomputed'] for r in trace)
        for trace in (left, right)]
    for i, (a, b) in enumerate(zip(left, right)):
        changes = {k: [a[k], b[k]] for k in a if a[k] != b[k]}
        if changes:
            report['first_difference'] = dict(boundary=i, changes=changes,
                stop_decisions=[float(a['score']) >= float(a['threshold']),
                                float(b['score']) >= float(b['threshold'])])
            break
    else:
        report['first_difference'] = None
    for label, fields in [('input', ['node', 'completed'] + [f'state{j}' for j in range(17)]),
                          ('decision', [])]:
        report['first_'+label+'_difference'] = None
        for i, (a, b) in enumerate(zip(left, right)):
            stops = [float(r['score']) >= float(r['threshold']) for r in (a, b)]
            changes = {k: [a[k], b[k]] for k in fields if a[k] != b[k]}
            if changes or (label == 'decision' and stops[0] != stops[1]):
                report['first_'+label+'_difference'] = dict(boundary=i,
                    changes=changes, scores=[a['score'], b['score']],
                    thresholds=[a['threshold'], b['threshold']], stop_decisions=stops)
                break
    return report


def diagnose(w):
    context = json.loads((w/'SERVER_CONTEXT.json').read_text())
    for rel, digest in context['source_hashes'].items():
        with (w/rel).open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError('Frozen source differs: '+rel)
    sys.path.insert(0, str(w/'experiments/p0_old_models_rerun'))
    from common import catalog
    c = catalog('sift100m')
    base = w/'experiments/p0_old_models_expanded/sift100m'
    parent = c['parent_dir']
    templates = json.loads((parent/'build.commands.json').read_text())
    template = next(cmd for cmd in templates
                    if str(parent/'search.cpp') in cmd and '-c' not in cmd)
    out = Path(tempfile.mkdtemp(prefix='et_trace_diagnostic_', dir=w))
    print('诊断目录：', out, flush=True)
    source = out/'search_trace.cpp'
    source.write_text(instrument((w/'experiments/p0_old_models_rerun/search.cpp').read_text()))
    command = list(template)
    command[command.index(str(parent/'search.cpp'))] = str(source)
    command[command.index('-o')+1] = str(out/'search_trace')
    (out/'build.command.json').write_text(json.dumps(command, indent=2)+'\n')
    print('编译独立诊断程序……', flush=True)
    with (out/'build.log').open('w') as log:
        subprocess.run(command, cwd=w, stdout=log, stderr=subprocess.STDOUT, check=True)
    traces = []
    summaries = []
    for rep in (6, 7):
        candidates = [p for p in (base/'logs').glob(f'test_r{rep}_*.configs')
                      if any(line.split() and line.split()[0] == KEY
                             for line in p.read_text().splitlines())]
        if len(candidates) != 1:
            raise ValueError('Expected one original batch: '+str(candidates))
        batch = out/candidates[0].name
        batch.write_bytes(candidates[0].read_bytes())
        dest = out/f'replay_r{rep}'
        dest.mkdir()
        print('开始追踪原批次', rep, flush=True)
        cmd = [out/'search_trace', c['index_path'],
               parent/'data'/f"test.{c['query_ext']}", parent/'data/test.gt',
               parent/'models', c['vector_path'], batch, dest, str(rep)]
        with (dest/'console.log').open('w') as log:
            subprocess.run(list(map(str, cmd)), cwd=w, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        row = next(r for r in rows(dest/f'{KEY}_r{rep}.csv')
                   if r['query'] == str(QUERY))
        original = next(r for r in rows(base/'runs/test'/f'{KEY}_r{rep}.csv')
                        if r['query'] == str(QUERY))
        fields = ['hits','reads','expansions','skips','et_evaluated','et_stopped','io_rounds']
        summary = dict(batch=rep, original={k: original[k] for k in fields},
                       traced={k: row[k] for k in fields})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        summaries.append(summary)
        traces.append(rows(dest/f'{KEY}_r{rep}.et.csv'))
    result = dict(query=QUERY, key=KEY, runs=summaries,
                  comparison=compare(*traces),
                  note='Diagnostic only; instrumentation may alter the original behaviour.')
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print('ET追踪对比：', json.dumps(result['comparison'], ensure_ascii=False), flush=True)
    print('保存于：', out/'comparison.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    w = args.workspace.resolve()
    if not (w/'SERVER_CONTEXT.json').is_file():
        parser.error('SERVER_CONTEXT.json not found')
    with (w/'server.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        diagnose(w)


if __name__ == '__main__':
    main()
