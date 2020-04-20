#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
bins = np.arange(0,110,10)
fig, ax = plt.subplots()
ax.hist(student_grades, bins = bins , edgecolor = 'black')
ax.set_xlim([0,100])
ax.set_xticks(bins)
ax.set_ylim([0,30])

ax.set_xlabel('Grades', fontsize=15)
ax.set_ylabel('Number of Students', fontsize=15)
ax.set_title("Project A")

plt.show()
