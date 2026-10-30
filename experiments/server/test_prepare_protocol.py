"""Protocol retry checks; import the helper without launching calibration."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

source = Path(__file__).resolve().parents[1]/'p0_old_models_rerun/prepare.py'
tree = ast.parse(source.read_text())
helper = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='freeze_protocol')
namespace = {'json':json,'dump':lambda p,v:p.write_text(json.dumps(v))}
exec(compile(ast.Module(body=[helper],type_ignores=[]),str(source),'exec'),namespace)
freeze = namespace['freeze_protocol']


class ProtocolRetryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'PROTOCOL.json'

    def tearDown(self):
        self.temp.cleanup()

    def test_same_protocol_can_be_prepared_twice(self):
        value={'P0_B':{'heuristic_fraction_pairs':[(1.,-1),(.72,1.)],
                       'joint_pairs':[(0,0),(4,3)]},'source_hashes':{'search':'abc'}}
        first = freeze(self.path,value)
        original = self.path.read_bytes()
        second = freeze(self.path,value)
        self.assertEqual(first,second)
        self.assertEqual(original,self.path.read_bytes())

    def test_real_parameter_change_still_fails_without_overwrite(self):
        freeze(self.path,{'beam':4})
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError,'fields: beam'):
            freeze(self.path,{'beam':8})
        self.assertEqual(original,self.path.read_bytes())

    def test_program_hash_change_names_the_file_and_preserves_protocol(self):
        freeze(self.path,{'source_hashes':{'native/search':'old'}})
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError,'changed input/program: native/search'):
            freeze(self.path,{'source_hashes':{'native/search':'new'}})
        self.assertEqual(original,self.path.read_bytes())


if __name__=='__main__':
    unittest.main()
