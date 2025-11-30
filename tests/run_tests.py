#!/usr/bin/env python
"""
NexNet Test Runner.

Run all tests: python run_tests.py
Run specific: python run_tests.py activations losses layers
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_activations import run_activation_tests
from tests.test_losses import run_loss_tests
from tests.test_layers import run_layer_tests
from tests.test_optimizers import run_optimizer_tests
from tests.test_models import run_model_tests
from tests.test_utils import run_utils_tests
from tests.test_integration import run_integration_tests


def main():
    print("=" * 70)
    print("                    NexNet Test Suite")
    print("=" * 70)
    
    total_passed = 0
    total_failed = 0
    
    test_suites = {
        'activations': run_activation_tests,
        'losses': run_loss_tests,
        'layers': run_layer_tests,
        'optimizers': run_optimizer_tests,
        'models': run_model_tests,
        'utils': run_utils_tests,
        'integration': run_integration_tests,
    }
    
    if len(sys.argv) > 1:
        suites_to_run = sys.argv[1:]
    else:
        suites_to_run = test_suites.keys()
    
    for suite_name in suites_to_run:
        if suite_name in test_suites:
            print(f"\n{'=' * 70}")
            passed, failed = test_suites[suite_name]()
            total_passed += passed
            total_failed += failed
        else:
            print(f"Unknown test suite: {suite_name}")
            print(f"Available suites: {', '.join(test_suites.keys())}")
    
    print("\n" + "=" * 70)
    print(f"                    FINAL RESULTS")
    print("=" * 70)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    print("=" * 70)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
