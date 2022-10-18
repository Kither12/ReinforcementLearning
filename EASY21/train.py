from easy21Env import *
from MonteCarloControl import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



# max_epoches = 1000000

# value = np.zeros((2, 22, 11))
# counter = np.zeros((2, 22, 11))
# for i in range(max_epoches) :
# 	value, counter = MonteCarlo(value, counter, 100)
# np.save("montecarlo.npy",value)
value = np.load("montecarlo.npy")
optimal_value = np.max(value, axis = 0)

ax = plt.axes(projection='3d')

x, y, z = [], [], []
for i in range(21) :
	for j in range(10) :
		x.append(i)
		y.append(j)
		z.append(optimal_value[i][j])

ax.plot_trisurf(y, x, z, cmap='viridis', edgecolor='none')
ax.set_ylabel("player starting card")
ax.set_xlabel("dealer current sum")
ax.set_zlabel("value of state")
plt.savefig('montecarlo.pdf')
plt.show()

