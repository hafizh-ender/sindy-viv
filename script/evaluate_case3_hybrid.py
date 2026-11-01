"""Evaluate saved Case 1/2 lift dynamics with saved Case 3 motion equations."""

import argparse

from sindy_viv.case_bryan.evaluate_sindy.case3_hybrid import evaluate_case3_hybrid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case_12_model', required=True,
                        help='Case 1/2 model directory, training.json, or model-info JSON.')
    parser.add_argument('--case_3_model', required=True,
                        help='Case 3 model directory, training.json, or manual model-info JSON.')
    parser.add_argument('--derivative_window_cases', nargs='+', required=True)
    parser.add_argument('--integral_window_case', required=True)
    parser.add_argument('--spectral_window_cases', nargs='+',
                        help='Defaults to all derivative windows; must be a subset of them.')
    parser.add_argument('--vr_list', nargs='+', type=int,
                        help='Defaults to the Case 1/2 training VR selection.')
    parser.add_argument('--lpd_list', nargs='+', type=float)
    parser.add_argument('--phase_comparison', action='store_true')
    parser.add_argument('--method', default='RK45')
    parser.add_argument('--max_state_factor', type=float, default=10.)
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--atol', type=float, default=1e-9)
    return parser.parse_args()


def main():
    result = evaluate_case3_hybrid(**vars(parse_args()))
    if result['failures']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    main()
