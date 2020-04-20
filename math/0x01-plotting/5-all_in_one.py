#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure()
fig.suptitle("All in One")
axes = plt.GridSpec(3, 2, wspace=0.4, hspace=1.2)
print(dir(fig))

plt.subplot(axes[0,0]).plot(np.arange(11), np.arange(0, 11) ** 3)
plt.subplot(axes[0,0]).set_xlim([0,10])

plt.subplot(axes[0,1]).scatter(x1, y1, c='magenta')
plt.subplot(axes[0,1]).set_xlabel('Height (in)', fontsize='x-small')
plt.subplot(axes[0,1]).set_ylabel('Weight (lbs)', fontsize='x-small')
plt.subplot(axes[0,1]).set_title("Men's Height vs Weight", fontsize='x-small')
plt.subplot(axes[1,0]).plot(x2,y2)
plt.subplot(axes[1,0]).set_xlim([0,28650])
plt.subplot(axes[1,0]).set_yscale('log')
plt.subplot(axes[1,0]).set_xlabel('Time (years)', fontsize='x-small')
plt.subplot(axes[1,0]).set_ylabel('Fraction Remaining', fontsize='x-small')
plt.subplot(axes[1,0]).set_title("Exponential Decay of C-14", fontsize='x-small')

line1, = plt.subplot(axes[1,1]).plot(x3,y31, 'k--', c='red')
line2, = plt.subplot(axes[1,1]).plot(x3,y32, c='green')

plt.subplot(axes[1,1]).set_xlim([0,20000])
plt.subplot(axes[1,1]).set_ylim([0,1])
plt.subplot(axes[1,1]).set_xlabel('Time (years)', fontsize='x-small')
plt.subplot(axes[1,1]).set_ylabel('Fraction Remaining', fontsize='x-small')
plt.subplot(axes[1,1]).set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
plt.subplot(axes[1,1]).legend([line1, line2], ['C-14', 'Ra-226'], loc = 'upper right', prop={'size': 8})

bins = np.arange(0,110,10)
plt.subplot(axes[2,0:]).hist(student_grades, bins = bins , edgecolor = 'black')
plt.subplot(axes[2,0:]).set_xlim([0,100])
plt.subplot(axes[2,0:]).set_xticks(bins)
plt.subplot(axes[2,0:]).set_ylim([0,30])
plt.subplot(axes[2,0:]).set_xlabel('Grades', fontsize='x-small')
plt.subplot(axes[2,0:]).set_ylabel('Number of Students', fontsize='x-small')
plt.subplot(axes[2,0:]).set_title("Project A", fontsize='x-small')

fig.align_labels()
plt.show()
