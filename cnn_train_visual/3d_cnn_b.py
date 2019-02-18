"""
========================================
Create 2D bar graphs in different planes
========================================

Demonstrates making a 3D plot which has 2D bar graphs projected onto
planes y=0, y=1, etc.
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c, z in zip(['c', 'r', 'g', 'b', 'y'], [5, 4, 3, 2, 1]):
    xs = np.arange(44)
    i = np.random.randint(0, np.shape(y_test)[0])
    ys_t = y_test[i]
#    ys_p = y_pred_w_noise[i]
    ys_p = y_pred[i]
    
    ax.plot(xs, ys_t, zs=z, zdir='y', alpha=0.8)
    ax.plot(xs, ys_p, zs=z, zdir='y', alpha=0.8, linestyle='dashed')
    
ax.set_xlabel('Order')
ax.set_ylabel('Sphere Index')
ax.set_zlabel('B Coefficient')
ax.grid(False)

plt.show()

