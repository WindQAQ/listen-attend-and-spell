import sys
import numpy as np


s = set()

f = np.load(sys.argv[1]).item()

for line in f.values():
    s.update(line)

d = sorted(list(s))

with open(sys.argv[2], 'w') as f:
    print('\n'.join(d), file=f)
