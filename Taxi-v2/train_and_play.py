import gym
import numpy as np
import random


env = gym.make('Taxi-v2')
action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

total_episodes = 50000
total_test_episodes = 2
max_steps = 99

learning_rate = 0.7
gamma = 0.618

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01


for episode in range(total_episodes):
	state = env.reset()
	done = False

	for step in range(max_steps):
		exp_exp_tradeoff = random.uniform(0,1)

		if exp_exp_tradeoff > epsilon:
			action = np.argmax(qtable[state,:])
		else:
			action = env.action_space.sample()

		new_state, reward, done, info = env.step(action)

		qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.argmax(qtable[new_state, :]) - qtable[state, action])

		state = new_state

		if done == True:
			break

	epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


rewards = []
max_steps = 40

try:
	np.save('qtable', qtable)
	print('Q-Table salva')
except:
	print('*****************************')
	print('**  Erro ao salvar qtable  **')
	print('*****************************')

for episode in range(total_test_episodes):
	state = env.reset()
	done = False
	total_rewards = 0

	print('***************************')
	print('Episode', episode)

	for step in range(max_steps):
		env.render()

		action = np.argmax(qtable[state,:])

		state, reward, done, info = env.step(action)

		total_rewards += reward

		if done:
			rewards.append(total_rewards)
			print('Score:', total_rewards)
			break

env.close()
print('Score over time: ', sum(rewards)/total_test_episodes)
