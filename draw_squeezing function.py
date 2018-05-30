import pylab
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)  # 100 linearly spaced numbers
z = 1.7159 * np.tanh((2 / 3) * x)

pylab.plot(x, z)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
pylab.show()  # show the plot