#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
people = ['Farrah', 'Fred', 'Felicia']
ind = [x for x, _ in enumerate(people)]

apples = fruit[0, :]
bananas = fruit[1, :]
oranges = fruit[2, :]
peaches = fruit[3, :]

plt.bar(people, apples,  width=0.5, label='apples', color='red')
plt.bar(people, bananas,  width=0.5, label='bananas', color='yellow', bottom=apples)
plt.bar(people, oranges,  width=0.5, label='oranges', color='#ff8000', bottom=apples+bananas)
plt.bar(people, peaches,  width=0.5, label='peaches', color='#ffe5b4', bottom=apples+bananas+oranges)

 
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.yticks(np.arange(0, 81, 10))
plt.legend(loc="upper right")

plt.show()