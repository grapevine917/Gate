#!/usr/bin/env python3
"""Prepare a fresh, isolated server run; never import historical timings.

Run --help and read docs/deployment/SERVER_RUNBOOK.md before transferring assets.
Historical scripts remain untouched; adaptations are confined to a new workspace.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
PARENTS = {'sift1m': 'main_comparison', 'deep1m': 'deep_p0', 'sift100m': 'sift100m_p0'}
EXT = {'sift1m': 'fbin', 'deep1m': 'fbin', 'sift100m': 'u8bin'}
INDEX = {'sift1m': 'experiments/sift1m/index', 'deep1m': 'experiments/deep1m/index',
         'sift100m': 'experiments/sift100m_p0/index/ann'}
CALIBRATION = {'sift1m': 'experiments/sift1m_p0a_L100_train_compare/data/calibration.prune.bin',
               'deep1m': 'experiments/deep_p0/data/validation.prune.bin',
               'sift100m': 'experiments/sift100m_p0/data/validation.prune.bin'}
RERUN = 'experiments/p0_old_models_rerun'
EXPANDED = 'experiments/p0_old_models_expanded'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(command, **kwargs):
    print('+', shlex.join(list(map(str, command))), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def artifact_paths(dataset):
    parent = 'experiments/'+PARENTS[dataset]
    paths = [f'{parent}/models/{name}.bin' for name in
             ['lr', 'state', 'xgb', 'et', 'native_check', 'et_check']]
    paths += [f'{parent}/data/{split}.{ext}' for split in ['validation', 'test']
              for ext in [EXT[dataset], 'gt']]
    paths += [f'{parent}/data/{split}_ids.npy' for split in ['train', 'validation', 'test']]
    paths += [parent+'/TRAIN_COMPLETE.json', CALIBRATION[dataset]]
    ref = 'DiskANN_L100_r0.ids' if dataset == 'sift1m' else 'DiskANN_L100_p0_r0.ids'
    paths += [parent+'/runs/validation/'+ref]
    paths += [INDEX[dataset]+s for s in ['_disk.index', '_pq_compressed.bin', '_pq_pivots.bin']]
    return paths


def asset_report(asset_root):
    missing = []
    total = 0
    records = []
    for ds in PARENTS:
        for relative in artifact_paths(ds):
            p = asset_root/relative
            if not p.is_file():
                missing.append(relative)
            else:
                total += p.stat().st_size
                records.append({'dataset': ds, 'path': relative, 'bytes': p.stat().st_size})
        # Preserve existing optional runtime auxiliaries, not index-build shards.
        for suffix in ['_disk.index_centroids.bin', '_disk.index_medoids.bin',
                       '_pq_pivots.bin_rotation_matrix.bin', '_disk.index_pq_pivots.bin']:
            p = asset_root/(INDEX[ds]+suffix)
            if p.is_file():
                total += p.stat().st_size
                records.append({'dataset': ds, 'path': str(p.relative_to(asset_root)),
                                'bytes': p.stat().st_size, 'optional_present': True})
    return {'asset_root': str(asset_root), 'missing': missing, 'files': records,
            'bytes': total, 'note': 'No old validation times, test runs or COMPLETE files are imported.'}


def ensure_fresh_path(workspace, assets):
    if workspace.exists():
        raise ValueError('Workspace already exists; choose a NEW path. Use later stages to resume.')
    if workspace == REPO or REPO in workspace.parents or assets in workspace.parents:
        raise ValueError('Workspace must be outside the code repository and asset tree.')


def create(workspace, assets):
    ensure_fresh_path(workspace, assets)
    report = asset_report(assets)
    if report['missing']:
        raise ValueError('Missing assets:\n'+'\n'.join(report['missing']))
    expected = json.loads((REPO/'experiments/server/assets_manifest.json').read_text())
    for item in expected['files']:
        p = assets/item['path']
        if not p.is_file() or p.stat().st_size != item['bytes']:
            raise ValueError('Missing asset or unexpected size: '+item['path'])
        if 'sha256' in item and sha(p) != item['sha256']:
            raise ValueError('Published asset hash mismatch: '+item['path'])
        if 'header4096_sha256' in item:
            with p.open('rb') as f:
                if hashlib.sha256(f.read(4096)).hexdigest() != item['header4096_sha256']:
                    raise ValueError('Published index header mismatch: '+item['path'])
    workspace.mkdir(parents=True)
    # Only source is copied. No previous benchmark metadata/runs are copied.
    for folder in [RERUN, EXPANDED]:
        dest = workspace/folder
        dest.mkdir(parents=True)
        for src in (REPO/folder).iterdir():
            if src.is_file() and src.suffix in ['.py', '.cpp']:
                shutil.copy2(src, dest/src.name)
    for ds, parent in PARENTS.items():
        dest = workspace/'experiments'/parent
        for relative in artifact_paths(ds):
            # Index/calibration paths are absolute in the private catalog below.
            if not relative.startswith('experiments/'+parent+'/'):
                continue
            if relative.startswith(INDEX[ds]):
                continue
            target = workspace/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            target.symlink_to(assets/relative)
        (dest/'native').mkdir(exist_ok=True)
        for src in (REPO/'experiments'/parent/'native').iterdir():
            if src.is_file() and src.suffix in ['.cpp', '.h']:
                shutil.copy2(src, dest/'native'/src.name)
    # Existing modules derive ROOT from their location, so all output is isolated.
    catalog = workspace/RERUN/'common.py'
    with catalog.open('a') as f:
        f.write('\n# Server-only paths, appended in the isolated workspace.\n')
        f.write('for _ds, _p in '+repr({ds: {'index': str(assets/INDEX[ds]),
                'calibration': str(assets/CALIBRATION[ds]), 'vectors': ''} for ds in PARENTS})+'.items():\n')
        f.write(' CATALOG[_ds].update(_p)\n')
    # A symlinked venv Python loses its original pyvenv.cfg when invoked from
    # another directory. Exec the original path so dependencies remain visible.
    venv = workspace/'.venv/bin'
    venv.mkdir(parents=True)
    launcher = venv/'python'
    launcher.write_text('#!/bin/sh\nexec '+shlex.quote(str(Path(sys.executable).absolute()))+' "$@"\n')
    launcher.chmod(0o755)
    # Do not mislabel a native Linux rerun as a WSL measurement in new reports.
    for relative in [RERUN+'/report.py', RERUN+'/prepare.py']:
        p = workspace/relative
        s = p.read_text().replace('WSL客户机块层', '本次运行环境的Linux块层')
        p.write_text(s)
    code = {str(p.relative_to(workspace)): sha(p) for p in workspace.rglob('*')
            if p.is_file() and not p.is_symlink() and p.suffix in ['.py', '.cpp', '.h']}
    # Large index files remain external; record sizes/head hashes once rather
    # than reading tens of GB during a benchmark. Model/query hashes are exact.
    files = []
    for item in report['files']:
        p = assets/item['path']
        row = dict(item)
        if item['path'].startswith(tuple(INDEX.values())):
            with p.open('rb') as f:
                row['header4096_sha256'] = hashlib.sha256(f.read(4096)).hexdigest()
        else:
            row['sha256'] = sha(p)
        files.append(row)
    dump(workspace/'SERVER_CONTEXT.json', {'repo': str(REPO), 'asset_root': str(assets),
         'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'python': sys.version,
         'platform': platform.platform(), 'source_hashes': code, 'assets': files,
         'entrypoint_sha256': sha(Path(__file__)),
         'native_verifier_sha256': sha(REPO/'experiments/server/verify_state.cpp'),
         'adaptations': ['isolated source-only copy', 'absolute index/calibration catalog',
                         'launcher for the new interpreter', 'environment wording in fresh reports'],
         'note': 'Previously observed queries; frozen historical weights. Server validation is rerun.'})
    print('Created', workspace, flush=True)


def check_workspace(workspace):
    context = json.loads((workspace/'SERVER_CONTEXT.json').read_text())
    for field, path in [('entrypoint_sha256', Path(__file__)),
                        ('native_verifier_sha256', REPO/'experiments/server/verify_state.cpp')]:
        if field in context and sha(path) != context[field]:
            raise ValueError('Server driver changed since workspace creation: '+str(path))
    for relative, digest in context['source_hashes'].items():
        if sha(workspace/relative) != digest:
            raise ValueError('Workspace source changed: '+relative)
    assets = Path(context['asset_root'])
    for item in context['assets']:
        p = assets/item['path']
        if p.stat().st_size != item['bytes']:
            raise ValueError('Asset size changed: '+item['path'])
        if 'sha256' in item and sha(p) != item['sha256']:
            raise ValueError('Asset hash changed: '+item['path'])
        if 'header4096_sha256' in item:
            with p.open('rb') as f:
                if hashlib.sha256(f.read(4096)).hexdigest() != item['header4096_sha256']:
                    raise ValueError('Index header changed: '+item['path'])
    return context


def build(workspace, args):
    diskann = args.diskann.resolve()
    base = args.base_build.resolve() if args.base_build else workspace/'build_diskann'
    if not args.base_build:
        command = ['cmake', '-S', diskann, '-B', base, '-G', 'Unix Makefiles',
                   '-DCMAKE_BUILD_TYPE=Release', '-DPYBIND=OFF', '-DRESTAPI=OFF']
        for key in ['MKL_PATH', 'MKL_INCLUDE_PATH', 'OMP_PATH']:
            value = getattr(args, key.lower())
            if value:
                command.append('-D'+key+'='+str(Path(value).resolve()))
        run(command)
        run(['cmake', '--build', base, '--target', 'diskann', '-j', args.jobs])
    flags = (base/'src/CMakeFiles/diskann.dir/flags.make').read_text()
    opts = []
    for key in ['CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS']:
        opts += shlex.split(next(x.split('=', 1)[1] for x in flags.splitlines()
                               if x.startswith(key+' =')))
    link = shlex.split((base/'apps/CMakeFiles/collect_io_labels.dir/link.txt').read_text())
    # Libraries come after the output filename in CMake's generated link line.
    libs = link[link.index('-o')+2:]
    if not any('libdiskann.a' in x for x in libs):
        raise ValueError('Expected a static DiskANN library in generated link flags.')
    compiler = shutil.which('g++')
    if not compiler:
        raise ValueError('g++ is required')
    commands = {}
    for ds in ([args.dataset] if args.dataset else PARENTS):
        parent = workspace/'experiments'/PARENTS[ds]
        native = parent/'native'
        library = native/'libdiskann.a'
        obj = native/'pq_flash_index.cpp.o'
        run([compiler, '-I'+str(native), *opts, '-c', native/'pq_flash_index.cpp', '-o', obj])
        shutil.copy2(base/'src/libdiskann.a', library)
        run(['ar', 'r', library, obj])
        local_libs = [str(library) if x.endswith('libdiskann.a') else x for x in libs]
        # Resolve relative paths from the generated apps link directory.
        local_libs = [str((base/'apps'/x).resolve()) if not x.startswith('-') and not Path(x).is_absolute()
                      else x for x in local_libs]
        local_libs += ['-latomic', '-lzstd']
        opts_ds = (['-DQUERY_U8'] if ds == 'sift100m' else [])+['-I'+str(native), *opts]
        output = workspace/RERUN/ds/'native'
        output.mkdir(parents=True, exist_ok=True)
        for source, name in [(workspace/RERUN/'search.cpp', 'search'),
                             (workspace/RERUN/'calibrate.cpp', 'calibrate'),
                             (REPO/'experiments/server/verify_state.cpp', 'verify_state')]:
            run([compiler, *opts_ds, source, '-o', output/name, *local_libs])
        # prepare.py identifies parent/search.cpp then substitutes calibrate.cpp.
        template = [compiler, *opts_ds, str(parent/'search.cpp'), '-o', str(native/'search'), *local_libs]
        dump(parent/'build.commands.json', [template])
        result = run([output/'verify_state', parent/'models', 96 if ds == 'deep1m' else 128],
                     capture_output=True, text=True)
        verification = json.loads(result.stdout)
        dump(output/'native_score_verification.json', verification)
        commands[ds] = {'native_scores': verification, 'search_sha256': sha(output/'search'),
                        'library_sha256': sha(library)}
    old = json.loads((workspace/'SERVER_BUILD.json').read_text()) if (workspace/'SERVER_BUILD.json').exists() else {}
    old.update(commands)
    dump(workspace/'SERVER_BUILD.json', old)


def stage(workspace, name):
    python = workspace/'.venv/bin/python'
    if name == 'prepare':
        built = json.loads((workspace/'SERVER_BUILD.json').read_text())
        if set(built) != set(PARENTS):
            raise ValueError('Build and verify all three datasets before prepare.')
        run([python, workspace/RERUN/'prepare.py'], cwd=workspace)
        run([python, workspace/RERUN/'index_lock.py'], cwd=workspace)
        return
    if name == 'expanded':
        for ds in PARENTS:
            if not (workspace/RERUN/ds/'SMOKE_COMPLETE.json').exists():
                raise ValueError('Run smoke successfully before expanded: '+ds)
        # Re-evaluate the original 64 validation configurations on THIS machine
        # before validation-local fine tuning; never import laptop timing JSON.
        for ds in PARENTS:
            command = ('import sys;sys.path.insert(0,'+repr(str(workspace/RERUN))+');'
                       'from measure import Experiment;from common import dump;'
                       'e=Experiment('+repr(ds)+');e.audit_sources();'
                       "dump(e.O/'validation.json',e.load('validation',e.configs))")
            run([python, '-c', command], cwd=workspace)
        run([python, workspace/EXPANDED/'run.py'], cwd=workspace)
        return
    for ds in PARENTS:
        if name == 'p0a' and not (workspace/RERUN/ds/'SMOKE_COMPLETE.json').exists():
            raise ValueError('Run smoke successfully before p0a: '+ds)
        run([python, workspace/RERUN/'worker.py', ds, name], cwd=workspace)
    if name == 'p0a':
        run([python, workspace/RERUN/'report.py', 'p0a'], cwd=workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'create', 'build', 'prepare', 'smoke', 'p0a', 'expanded'])
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--diskann', type=Path, default=REPO/'DiskANN')
    parser.add_argument('--base-build', type=Path, help='Reuse a base build on THIS machine only; default is a fresh CMake build.')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--dataset', choices=list(PARENTS), help='Only for build; remaining stages require all three datasets.')
    for name in ['mkl-path', 'mkl-include-path', 'omp-path']:
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.dataset and args.action != 'build':
        parser.error('--dataset is supported only for build')
    if args.action in ['inspect', 'create'] and not args.asset_root:
        parser.error('--asset-root is required')
    if args.action != 'inspect' and not args.workspace:
        parser.error('--workspace is required')
    if args.action == 'inspect':
        report = asset_report(args.asset_root.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report['missing'] else 0
    workspace = args.workspace.resolve()
    if args.action == 'create':
        create(workspace, args.asset_root.resolve())
        return 0
    check_workspace(workspace)
    # Lock covers build, validation, timing and plotting, preventing duplicate
    # stage launches in one workspace; do not benchmark workspaces concurrently.
    with (workspace/'server.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'build':
            build(workspace, args)
        else:
            stage(workspace, args.action)
    return 0


if __name__ == '__main__':
    sys.exit(main())
