#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

x = np.arange(11)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlim([0,10])
plt.show()
