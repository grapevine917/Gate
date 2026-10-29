"""Asset/isolation safeguards. No graph search, network or real data required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

spec = importlib.util.spec_from_file_location('server_benchmark', Path(__file__).with_name('benchmark.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class ServerWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.repo = self.base/'code'
        self.assets = self.base/'assets'
        self.output = self.base/'runs'/'fresh'
        self.previous = b.REPO
        b.REPO = self.repo
        for folder in [b.RERUN, b.EXPANDED, 'experiments/server']:
            (self.repo/folder).mkdir(parents=True)
        (self.repo/b.RERUN/'common.py').write_text('CATALOG = {}\n')
        (self.repo/b.RERUN/'report.py').write_text('# WSL客户机块层\n')
        (self.repo/b.RERUN/'prepare.py').write_text('# prepare\n')
        (self.repo/b.RERUN/'COMPLETE.json').write_text('{"complete":true}')
        (self.repo/'experiments/server/verify_state.cpp').write_text('// verify\n')
        for ds, parent in b.PARENTS.items():
            native = self.repo/'experiments'/parent/'native'
            native.mkdir(parents=True)
            (native/'online_gate.h').write_text('// frozen source\n')
            for relative in b.artifact_paths(ds):
                f = self.assets/relative
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b'fixture')
        report = b.asset_report(self.assets)
        for row in report['files']:
            row['sha256'] = b.sha(self.assets/row['path'])
        b.dump(self.repo/'experiments/server/assets_manifest.json', report)

    def tearDown(self):
        b.REPO = self.previous
        self.temp.cleanup()

    def test_missing_assets_are_reported(self):
        missing = self.assets/b.artifact_paths('deep1m')[0]
        missing.unlink()
        self.assertIn(str(missing.relative_to(self.assets)), b.asset_report(self.assets)['missing'])

    def test_isolated_create_does_not_import_completion_or_timings(self):
        b.create(self.output, self.assets)
        self.assertFalse((self.output/b.RERUN/'COMPLETE.json').exists())
        self.assertFalse((self.output/b.RERUN/'sift1m/validation.json').exists())
        model = self.output/'experiments/main_comparison/models/lr.bin'
        self.assertTrue(model.is_symlink())
        self.assertEqual(model.resolve(), self.assets/'experiments/main_comparison/models/lr.bin')
        self.assertTrue(b.check_workspace(self.output)['assets'])

    def test_no_overwrite_or_nested_workspace(self):
        self.output.mkdir(parents=True)
        with self.assertRaises(ValueError):
            b.create(self.output, self.assets)
        with self.assertRaises(ValueError):
            b.create(self.assets/'danger', self.assets)

    def test_bad_asset_hash_stops_before_creation(self):
        f = self.assets/b.artifact_paths('sift1m')[0]
        f.write_bytes(b'changed')  # same size; only content hash can catch this
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            b.create(self.output, self.assets)
        self.assertFalse(self.output.exists())

    def test_source_mutation_is_rejected_before_resume(self):
        b.create(self.output, self.assets)
        (self.output/b.RERUN/'common.py').write_text('# altered\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            b.check_workspace(self.output)

    def test_python_launcher_preserves_virtual_environment(self):
        b.create(self.output, self.assets)
        prefix = subprocess.check_output([self.output/'.venv/bin/python', '-c',
                                          'import sys;print(sys.prefix)'], text=True).strip()
        self.assertEqual(prefix, sys.prefix)


if __name__ == '__main__':
    unittest.main()
