#!/usr/bin/env python
import argparse
import locale
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
from run import read_tensor, run

parser = argparse.ArgumentParser(description='Test connx model')
parser.add_argument('--atol', type=float, default=1e-7, help='absolute tolerance')
parser.add_argument('--rtol', type=float, default=0.001, help='relative tolerance')
parser.add_argument('connx', type=str, default='./connx', help='connx executable file')
parser.add_argument('model', type=str, help='model directory')
parser.add_argument(
    'tests', type=str, metavar='test case', nargs="*",
    help='test case directories')


np.set_printoptions(suppress=True, linewidth=160, threshold=sys.maxsize)

locale.setlocale(locale.LC_ALL, '')

PASS = '\033[92m'
FAIL = '\033[91m'
END = '\033[0m'

args = parser.parse_args()

CONNX = args.connx
HOME = args.model

total = 0  # total time consumption

pass_count = 0
fail_count = 0

for path in Path(HOME).rglob('*.connx'):
    if args.tests and not any(tc in str(path) for tc in args.tests):
        continue

    dataset = glob(os.path.join(path.parent, 'test_data_set_*'))

    for data in dataset:
        input_paths = glob(os.path.join(data, 'input_*.data'))
        input_paths.sort()
        output_paths = glob(os.path.join(data, 'output_*.data'))
        output_paths.sort()

        name = path.parent.name
        print('# Test:', name, end=' ', flush=True)
        model_path = os.path.join(path.parent)

        start_time = time.time()
        outputs = run(CONNX, model_path, input_paths)
        end_time = time.time()

        is_failed = False

        if len(outputs) != len(output_paths):
            print('  Number of output count is different: inferenced: {}, reference: {}'
                  .format(len(outputs), len(output_paths)), flush=True)

            is_failed = True

        for idx, (output, output_path) in enumerate(zip(outputs, output_paths)):
            with open(output_path, 'rb') as io:
                ref = read_tensor(io)

            if output.shape != ref.shape:
                is_failed = True
                print('  shape of output[{}] is differ:'.format(idx))
                print('  ## Inferenced tensor')
                print(output.shape)
                print('  ## Reference tensor')
                print(ref.shape, flush=True)
                continue

            if not np.allclose(output, ref, atol=args.atol, rtol=args.rtol):
                is_failed = True
                print('  data of output[{}] is differ:'.format(idx))
                print('  ## Inferenced tensor')
                print(output)
                print('  ## Reference tensor')
                print(ref, flush=True)
                continue

        if is_failed:
            print(f'{FAIL}Failed{END}', flush=True)
            fail_count += 1
        else:
            dt = end_time - start_time
            total += dt
            print(f'{dt * 1000:n} ms {PASS}Passed{END}', flush=True)
            pass_count += 1


print(f'Time: {total * 1000:n} ms, '
      f'{PASS if fail_count == 0 else ""}PASS: {pass_count}{END if fail_count == 0 else ""}, '
      f'{FAIL if fail_count > 0 else ""}FAIL: {fail_count}{END if fail_count > 0 else ""}')

exit(1 if fail_count > 0 else 0)
