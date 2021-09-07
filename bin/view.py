import sys
import locale
import numpy as np
import matplotlib.pyplot as plt
from run import read_tensor


if len(sys.argv) < 2:
    print('Usage: {} [connx input.data path]'.format(sys.argv[0]))
    sys.exit(0)

locale.setlocale(locale.LC_ALL, '')

DATA_PATH = sys.argv[1]
input_data = None
with open(DATA_PATH, 'rb') as io:
    input_tensor = read_tensor(io).astype(np.uint8)
    b, c, h, w = input_tensor.shape

    img = input_tensor[0].reshape((h, w, c))


if img is not None:
    plt.imshow(img)
    plt.show()
