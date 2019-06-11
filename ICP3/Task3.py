import numpy as np
a = np.random.randint(1,20,10)
print(a)
maxElement = np.amax(a)
a[np.where(a == np.amax(a))] = 0
print(a)