import numpy as np

v = np.random.randint(1, 100, (3, 3))
print(v)
print(np.sort(v, axis=0))
print(np.argsort(v, axis=0))
