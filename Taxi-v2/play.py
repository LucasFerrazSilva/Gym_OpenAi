import numpy as np
import gym

env = gym.make('Taxi-v2')
qtable = np.load('qtable.npy')

state = env.reset()
score = 0

for step in range(99):
	print(step)
	env.render()

	action = np.argmax(qtable[state,:])

	state, reward, done, info = env.step(action)

	score += reward

	if done == True:
		print('Score:', score)
		break