import  numpy as np

scales = [25., 1., 1., 1., 1.]
c = 1.0
scales = [c * x for x in scales]
shapes = [4., 1., 1., 1., 1.]
shapes = [x / c for x in shapes]

shape, scale = shapes[0], scales[0]  # mean and dispersion
s = np.random.gamma(shape, scale, 10000)

print np.mean(s)
print np.std(s)
import matplotlib.pyplot as plt
import scipy.special as sps

count, bins, ignored = plt.hist(s, 100, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale) /
                     (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()