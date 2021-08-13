#!/usr/bin/env python3

import os
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print('Usage: {} [base dir] [license file]'.format(sys.argv[0]))
    sys.exit(0)

base_dir = sys.argv[1]
license_file = sys.argv[2]


def get_copyright(lines):
    start = -1
    end = -1
    copyright = False
    for i in range(min(20, len(lines))):
        line = lines[i].strip()

        if not copyright:
            if line.startswith('/*'):
                start = i
            elif 'Copyright' in line or 'copyright' in line:
                copyright = True
        else:
            if line.endswith('*/'):
                end = i
                break

    if start != -1 and end != -1 and copyright:
        return (start, end)
    else:
        return None


# read license
license = []
with open(str(license_file), 'r') as f:
    license = f.readlines()

# retrieve files
for path in Path(base_dir).rglob('*'):
    #print(path, path.name)
    if not os.path.isfile(str(path)):
        continue

    pos = None

    # Check the file has copyright term
    with open(str(path), 'r') as f:
        lines = f.readlines()

        pos = get_copyright(lines)

    # Relicense
    if pos is not None:
        print('Relicensing: {}'.format(str(path)))

        with open(str(path), 'w') as f:
            f.writelines(lines[:pos[0]])
            f.writelines(license)
            f.writelines(lines[pos[1] + 1:])
