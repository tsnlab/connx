#!/usr/bin/env python3

import sys

import numpy as np
from run import read_tensor


np.set_printoptions(suppress=True, linewidth=160, threshold=sys.maxsize)

if len(sys.argv) < 2:
    print('Usage: {} [data file]'.format(sys.argv[0]))
    sys.exit(0)

with open(sys.argv[1], 'rb') as io:
    tensor = read_tensor(io)
    print(tensor.shape)
    print(tensor)
