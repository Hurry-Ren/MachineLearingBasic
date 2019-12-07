import numpy as np

a = [0, 0, 1, 1, 0.5]
b = [1, 0, 0, 0.5, 0.5]
for i in range(len(a)):
    print(min(a[i], b[i]))
