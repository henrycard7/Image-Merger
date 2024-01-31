import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from pathlib import Path
from itertools import permutations
from multiprocessing import Pool

from MyHybridImages import myHybridImages

file_dir = Path(__file__).resolve().parent

im1 = cv2.imread(str('data/xiaohao2.png'))
im2 = cv2.imread(str('data/hansung3.png'))

def hybrid(sigs):
    sig1, sig2 = sigs

    test = myHybridImages(im1, sig1, im2, sig2)

    cv2.imwrite(str(file_dir/f'output/hybrid_{sig1}_{sig2}.png'), test)


# if __name__ == '__main__':
    # with Pool(3) as p:
        # p.map(hybrid, list(permutations(range(1, 21), 2)))

sig1, sig2 = 4, 1

test = myHybridImages(im1, sig1, im2, sig2)
cv2.imwrite(str(file_dir/f'hybrid_{sig1}_{sig2}.png'), test)