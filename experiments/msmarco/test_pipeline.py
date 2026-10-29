"""Small correctness fixtures only; these are not experiment results."""
import argparse
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.distance import cdist

spec=importlib.util.spec_from_file_location('pipeline',Path(__file__).with_name('pipeline.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PipelineTests(unittest.TestCase):
    def test_native_build_covers_consumers_before_running(self):
        with patch.object(m, 'run') as run:
            m.ensure_native(argparse.Namespace(stage='collect', out=Path('/tmp')))
            self.assertIn('collect_io_labels', run.call_args.args[0])
            run.reset_mock()
            m.ensure_native(argparse.Namespace(stage='evaluate', out=Path('/tmp')))
            self.assertIn('search_disk_index', run.call_args.args[0])
            self.assertIn('search_query_skip_cache', run.call_args.args[0])
            run.reset_mock()
            m.ensure_native(argparse.Namespace(stage='fit', out=Path('/tmp')))
            run.assert_not_called()

    def test_query_columns_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'queries.tsv';p.write_text('12\t \t example query \n13\tsecond\n')
            self.assertEqual(list(m.queries(p)),[{'id':'12','text':'example query'},{'id':'13','text':'second'}])
            p.write_text('12\tfirst\n12\tsecond\n')
            with self.assertRaises(ValueError):list(m.queries(p))

    def test_gt_matches_direct_distances_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);rng=np.random.default_rng(7)
            base=rng.normal(size=(41,8)).astype('<f4');q=rng.normal(size=(5,8)).astype('<f4')
            base[1]=base[0];q[0]=base[0]
            m.write(out/'base.fbin',base)
            for s in ['train','validation','test']:m.write(out/s/'query.fbin',q)
            m.truth(argparse.Namespace(out=out))
            expected=np.argsort(cdist(q,base,'sqeuclidean'),axis=1,kind='stable')[:,:10]
            np.testing.assert_array_equal(m.matrix(out/'test/gt.bin','<u4'),expected)
            with (out/'base.fbin').open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):m.matrix(out/'base.fbin')

    def test_fit_marks_use_all_observations_and_model_is_native_format(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);m.write(out/'base.fbin',np.zeros((20,8),'<f4'))
            for s in ['train','validation']:
                (out/s).mkdir(exist_ok=True)
                with (out/s/'collected_features.csv').open('w') as f, (out/s/'collected_candidates.csv').open('w') as g:
                    fw=csv.writer(f);cw=csv.writer(g)
                    fw.writerow(['query','node']+[f's{i}' for i in range(17)])
                    cw.writerow(['query','node','cache_miss','fresh','admitted','in_result','singleton_recall_ok','singleton_exact','singleton_reads','singleton_delta'])
                    for q in range(20):
                        for node in range(4):
                            x=np.random.default_rng(q*4+node).normal(size=17);x[10]=0
                            fw.writerow([q,node,*x])
                            # node 0 always low; node 1 has one high event;
                            # node 2 is in final results; node 3 is cached only.
                            cw.writerow([q,node,int(node!=3),1,int(node==1 and q==0),int(node==2),int(node%2==0),0,10,int(node%2==0)])
            m.fit(argparse.Namespace(out=out))
            payload=(out/'marks.bin').read_bytes()
            self.assertEqual(payload[:8],b'NDMARK01')
            bits=np.unpackbits(np.frombuffer(payload[12:],dtype='u1'),bitorder='little')[:20]
            self.assertEqual(np.flatnonzero(bits).tolist(),[0])
            model=(out/'lr.bin').read_bytes()
            self.assertEqual(model[:8],b'LRGATE01')
            self.assertEqual(len(model),12+(17*3+1)*4)

if __name__=='__main__':unittest.main()
