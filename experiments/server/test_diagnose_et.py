"""Check that trace reports distinguish changed inputs from changed decisions."""
import unittest
from diagnose_et import compare


def row(score='0.9', state='1'):
    r = {f'state{i}': '1' for i in range(17)}
    r.update(node='123456789', completed='441', threshold='0.95',
             score=score, recomputed=score, boundary='0')
    r['state0'] = state
    return r


class TraceComparisonTests(unittest.TestCase):
    def test_same_trace(self):
        report = compare([row()], [row()])
        self.assertIsNone(report['first_difference'])
        self.assertIsNone(report['first_decision_difference'])

    def test_identical_input_different_stop(self):
        report = compare([row()], [row('0.99')])
        self.assertIsNone(report['first_input_difference'])
        self.assertEqual(report['first_decision_difference']['stop_decisions'], [False, True])

    def test_input_difference_without_changed_stop(self):
        report = compare([row()], [row(state='2')])
        self.assertEqual(report['first_input_difference']['changes'], {'state0': ['1', '2']})
        self.assertIsNone(report['first_decision_difference'])

    def test_extra_boundary_is_visible(self):
        self.assertEqual(compare([row()], [row(), row()])['boundaries'], [1, 2])

    def test_recomputed_score_disagreement_is_visible(self):
        report = compare([row()], [dict(row(), recomputed='0.99')])
        self.assertEqual(report['recomputed_mismatches'], [0, 1])


if __name__ == '__main__':
    unittest.main()
