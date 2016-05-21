import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 1000)
y = np.maximum(x, 0)

plt.figure(figsize=(8, 4))
plt.axis([-2, 2, -0.5, 2])
plt.plot(x, y, linewidth=2)
plt.xticks([0])
plt.yticks([0], fontsize='medium')
plt.xlabel('z', fontsize='medium')
plt.ylabel('g(z)=max{0,z}', fontsize='medium')
plt.savefig('report_images/relu.png', dpi=250, bbox_inches='tight')
plt.figure(figsize=(8, 4))

plt.xlabel('z', fontsize='medium')
plt.ylabel('g(z)=tanh(z)', fontsize='medium')
plt.plot(x, np.tanh(x), linewidth=2)
plt.xticks([0])
plt.yticks([0], fontsize='medium')
# plt.axis([-2,2,-0.5,2])
plt.savefig('report_images/tanh.png', dpi=250, bbox_inches='tight')

plt.show()
