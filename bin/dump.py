#!/usr/bin/env python3

import sys
from run import read_tensor


if len(sys.argv) < 2:
    print('Usage: {} [data file]'.format(sys.argv[0]))
    sys.exit(0)

with open(sys.argv[1], 'rb') as io:
    tensor = read_tensor(io)
    print(tensor)
