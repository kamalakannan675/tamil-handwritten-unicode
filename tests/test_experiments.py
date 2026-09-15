"""
test_experiments.py - Unit tests for experiment orchestration and comparison reports.
"""

import os
import json
import csv
import unittest
from scripts.run_experiments import compile_comparison_report

class TestExperiments(unittest.TestCase):
    def test_compile_comparison_report(self):
        """Verify model comparison compilation returns correct structure."""
        report = compile_comparison_report()
        self.assertIn('comparison_table', report)
        self.assertIn('evaluation_scope', report)
        self.assertTrue(len(report['comparison_table']) >= 1)

        first_row = report['comparison_table'][0]
        self.assertIn('model_name', first_row)
        self.assertIn('top1_accuracy_percent', first_row)
        self.assertIn('top3_accuracy_percent', first_row)
        self.assertIn('macro_f1', first_row)

    def test_comparison_csv_format(self):
        """Verify model_comparison.csv exists and has expected header."""
        csv_path = 'outputs/reports/model_comparison.csv'
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header[0], 'Experiment')
            self.assertEqual(header[5], 'Top-1 (%)')
            self.assertEqual(header[6], 'Top-3 (%)')

    def test_baseline_artifacts_preserved(self):
        """Verify baseline experiment directory and metrics are preserved."""
        baseline_metrics = 'outputs/reports/experiments/baseline/metrics.json'
        self.assertTrue(os.path.exists(baseline_metrics))
        with open(baseline_metrics, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['total_test_samples'], 28080)
        self.assertAlmostEqual(data['top1_accuracy_percent'], 71.84, places=1)

if __name__ == '__main__':
    unittest.main()
