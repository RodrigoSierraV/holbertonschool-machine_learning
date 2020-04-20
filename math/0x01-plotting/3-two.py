#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

fig, ax = plt.subplots()
line1, = ax.plot(x,y1, 'k--', c='red')
line2, = ax.plot(x,y2, c='green')

ax.set_xlim([0,20000])
ax.set_ylim([0,1])

ax.set_xlabel('Time (years)', fontsize=15)
ax.set_ylabel('Fraction Remaining', fontsize=15)
ax.set_title("Exponential Decay of Radioactive Elements")

plt.legend([line1, line2], ['C-14', 'Ra-226'], loc = 'upper right')

plt.show()
