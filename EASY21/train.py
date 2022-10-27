from easy21Env import *
from MonteCarloControl import *
from SARSA import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



# max_epoches = 1000000

# value = np.zeros((2, 22, 11))
# counter = np.zeros((2, 22, 11))
# for i in range(max_epoches) :
# 	value, counter = MonteCarlo(value, counter, 100)
# 	optimal_value = np.max(value, axis = 0)
# 	print(np.sum(optimal_value))
# np.save("montecarlo.npy",value)
# value = np.load("montecarlo.npy")

# ax = plt.axes(projection='3d')

# x, y, z = [], [], []
# for i in range(1, 22) :
# 	for j in range(1, 11) :
# 		x.append(i)
# 		y.append(j)
# 		z.append(optimal_value[i][j])

# ax.plot_trisurf(y, x, z, cmap='viridis', edgecolor='none')
# ax.set_ylabel("player starting card")
# ax.set_xlabel("dealer current sum")
# ax.set_zlabel("value of state")
# plt.savefig('montecarlo.pdf')
# plt.show()

# monte_value = np.max(np.load("montecarlo.npy"), axis = 0)


# x, y = [], []

# max_epoches = 1000000

# for alpha in range(0, 11, 1):
# 	value = np.zeros((2, 22, 11))
# 	counter = np.zeros((2, 22, 11))
# 	for i in range(max_epoches) :
# 		value, counter = SARSA(value, counter, alpha / 10, 100)
# 	optimal_value = np.max(value, axis = 0)
# 	MSE = np.sum(np.square(monte_value - optimal_value))
# 	y.append(MSE)
# 	x.append(alpha / 10)

# plt.plot(x, y)
# plt.savefig('MSE_over_lambda.pdf')
# plt.show()

# x, y, x2, y2 = [], [], [], []

# max_epoches = 1000000

# for alpha in [0, 1]:
# 	value = np.zeros((2, 22, 11))
# 	counter = np.zeros((2, 22, 11))
# 	for i in range(max_epoches) :
# 		value, counter = SARSA(value, counter, alpha, 100)
# 		if i % 50000 == 0 :
# 			optimal_value = np.max(value, axis = 0)
# 			MSE = np.sum(np.square(monte_value - optimal_value))
# 			if alpha == 1 :
# 				y.append(MSE)
# 				x.append(i)
# 			else :
# 				y2.append(MSE)
# 				x2.append(i)

# plt.plot(x, y, color = 'r')
# plt.plot(x2, y2, color = 'g')
# plt.savefig("MSE_vs_Lambda01")
# plt.show()


monte_value = np.load("montecarlo.npy")
rewards = 0
for i in range(1000000):
	env = Easy21Env()
	while not env.terminate:
		action = np.argmax(monte_value[:, env.player.points, env.dealer.points])
		reward = env.step(action)
		rewards += reward
print(rewards)

rewards = 0
for i in range(1000000):
	env = Easy21Env()
	while not env.terminate:
		action = np.random.randint(0, 2)
		reward = env.step(action)
		rewards += reward
print(rewards)

