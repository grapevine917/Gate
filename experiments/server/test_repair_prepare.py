"""A comparison-only repair must preserve frozen protocols and binaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import repair_prepare as repair


class RepairPrepareTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.repo = self.base/'code'
        self.workspace = self.base/'run'
        self.assets = self.base/'assets'
        self.assets.mkdir()
        self.source = self.repo/repair.RELATIVE
        self.source.parent.mkdir(parents=True)
        self.source.write_text('# corrected comparison\n')
        self.target = self.workspace/repair.RELATIVE
        self.target.parent.mkdir(parents=True)
        self.target.write_text('# original comparison\n')
        self.protocol = self.target.parent/'sift1m/PROTOCOL.json'
        self.protocol.parent.mkdir()
        self.protocol.write_text('{"beam":4,"source_hashes":{"search":"frozen"}}\n')
        self.binary = self.protocol.parent/'native/search'
        self.binary.parent.mkdir()
        self.binary.write_bytes(b'unchanged binary')
        repair.b.dump(self.workspace/'SERVER_CONTEXT.json',{
            'asset_root':str(self.assets),'assets':[],
            'source_hashes':{repair.RELATIVE:repair.b.sha(self.target)}})
        self.repo_patch = patch.object(repair.b,'REPO',self.repo)
        self.previous_patch = patch.object(repair,'PREDECESSOR',repair.b.sha(self.target))
        self.repo_patch.start();self.previous_patch.start()

    def tearDown(self):
        self.previous_patch.stop();self.repo_patch.stop();self.temp.cleanup()

    def test_repair_is_backed_up_idempotent_and_keeps_protocol_binary(self):
        protocol = self.protocol.read_bytes();binary = self.binary.read_bytes()
        original = self.target.read_bytes()
        repair.repair(self.workspace)
        self.assertEqual(self.target.read_text(),self.source.read_text())
        self.assertEqual(self.protocol.read_bytes(),protocol)
        self.assertEqual(self.binary.read_bytes(),binary)
        context = json.loads((self.workspace/'SERVER_CONTEXT.json').read_text())
        self.assertEqual((Path(context['repairs'][0]['backup'])/'prepare.py').read_bytes(),original)
        self.assertEqual(context['source_hashes'][repair.RELATIVE],repair.b.sha(self.target))
        repair.repair(self.workspace)
        self.assertEqual(len(list((self.workspace/'repair_backups').iterdir())),1)

    def test_repair_rejects_unrecognized_source_before_modifying(self):
        self.target.write_text('# unknown edits\n')
        context = json.loads((self.workspace/'SERVER_CONTEXT.json').read_text())
        context['source_hashes'][repair.RELATIVE] = repair.b.sha(self.target)
        repair.b.dump(self.workspace/'SERVER_CONTEXT.json',context)
        original = self.target.read_bytes()
        with self.assertRaisesRegex(ValueError,'Unrecognized'):
            repair.repair(self.workspace)
        self.assertEqual(self.target.read_bytes(),original)
        self.assertFalse((self.workspace/'repair_backups').exists())


if __name__=='__main__':
    unittest.main()
