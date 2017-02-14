"""
Created on 7:38 PM , 2/13/17, 2017

        by Tommy Tang
        
Project2 - test
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 10)
for i in range(1, 6):
    plt.plot(x, i * x + i, label='$y = {i}x + {i}$'.format(i=i))
plt.legend(loc='best')
plt.show()

plt.plot(x, x*x, label='Square')
plt.show()
