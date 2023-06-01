# fmt: off
import sys

package_path = '/Users/maxwuerfek/code/diss/spike_corr/build/lib.macosx-11.1-arm64-cpython-38'
sys.path.append(package_path)

import STTC
import numpy as np


st1 = np.array([2.1, 6, 10])
st2 = np.array([1, 2, 2.2, 5])
time = np.array([0, 11])
dt = 0.5

x = STTC.sttc(st1, st2, dt, time)
print(x) # 0.1550688802543270?

sts = [st1, st2]
STTC.tiling(sts)
