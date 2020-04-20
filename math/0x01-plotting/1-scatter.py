#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

fig, ax = plt.subplots()

ax.scatter(x, y, c='magenta')

ax.set_xlabel('Height (in)', fontsize=15)
ax.set_ylabel('Weight (lbs)', fontsize=15)
ax.set_title("Men's Height vs Weight")

plt.show()
