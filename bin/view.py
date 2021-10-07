import sys
import locale
import numpy as np
import matplotlib.pyplot as plt
from run import read_tensor


if len(sys.argv) < 2:
    print('Usage: {} [connx input.data path]'.format(sys.argv[0]))
    sys.exit(0)

locale.setlocale(locale.LC_ALL, '')

with open(sys.argv[1], 'rb') as io:
    input_tensor = read_tensor(io)
    input_tensor = input_tensor.astype(np.uint8)
    print(input_tensor.shape)
    print(np.min(input_tensor[0, 0]), np.max(input_tensor[0, 0]))
    print(np.min(input_tensor[0, 1]), np.max(input_tensor[0, 1]))
    print(np.min(input_tensor[0, 2]), np.max(input_tensor[0, 2]))
    b, c, h, w = input_tensor.shape

    img = input_tensor[0, 0].reshape((h, w, 1))


if img is not None:
    plt.imshow(img)
    plt.show()
