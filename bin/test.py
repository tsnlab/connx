import os
import sys
import time
import locale
from pathlib import Path
from glob import glob
import numpy as np
from run import run, read_tensor


np.set_printoptions(suppress=True, linewidth=160, threshold=sys.maxsize)

if len(sys.argv) < 3:
    print('Usage: {} [connx path] [connx home path] [[test case] ...]'.format(sys.argv[0]))
    sys.exit(0)

locale.setlocale(locale.LC_ALL, '')

PASS = '\033[92m'
FAIL = '\033[91m'
END = '\033[0m'

CONNX = sys.argv[1]
HOME = sys.argv[2]

total = 0  # total time consumption

pass_count = 0
fail_count = 0

for path in Path(HOME + '/test').rglob('*.connx'):
    if len(sys.argv) > 3:
        is_found = False

        for tc in sys.argv[3:]:
            if tc in str(path):
                is_found = True
                break

        if not is_found:
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

        is_passed = True

        if len(outputs) != len(output_paths):
            if is_passed:
                print(f'{FAIL}Failed{END}', flush=True)
                fail_count += 1

            print('  Number of output count is different: inferenced: {}, reference: {}'
                  .format(len(outputs), len(output_paths)), flush=True)

            is_passed = False

        for idx, (output, output_path) in enumerate(zip(outputs, output_paths)):
            with open(output_path, 'rb') as io:
                ref = read_tensor(io)

            if not np.allclose(output, ref, atol=1e-07, rtol=0.001):
                if is_passed:
                    print(f'{FAIL}Failed{END}', flush=True)
                    fail_count += 1

                print('  data of output[{}] is differ:'.format(idx))
                print('  ## Inferenced tensor')
                print(output)
                print('  ## Reference tensor')
                print(ref, flush=True)

                is_passed = False
                continue

        if is_passed:
            dt = end_time - start_time
            total += dt
            print(f'{dt * 1000:n} ms {PASS}Passed{END}', flush=True)
            pass_count += 1

print(f'Time: {total * 1000:n} ms, '
      f'{PASS if fail_count == 0 else ""}PASS: {pass_count}{END if fail_count == 0 else ""}, '
      f'{FAIL if fail_count > 0 else ""}FAIL: {fail_count}{END if fail_count > 0 else ""}')
