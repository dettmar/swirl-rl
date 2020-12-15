import sys
import datetime
import gym
import torch
from ACDir import ActorCriticDiscrete

env = gym.make('gym_swirl:swirl-v1')
env.seed(42)

print("env", env)

if __name__ == "__main__":

	torch.manual_seed(42)
	deg2rad = lambda x: x * 3.1415926536 / 180.
	rad2deg = lambda x: x * 180. / 3.1415926536

	actions = deg2rad(torch.tensor([-1, 0, 1]))
	actions_desc = ["<", "^", ">"]
	ac = ActorCriticDiscrete(num_inputs=1, num_actions=3)

	optimizer = torch.optim.Adam(ac.parameters(), lr=0.01, betas=(0.9, 0.999)) #lr=0.02
	epochs = 300
	measurements = 200
	timestep_size = 100 # 100*.2s*4deg/s = 80 deg (enough to see new behaviour)
	rewards = []
	angles = []
	use_means = False
	use_bonuses = True
	for i in range(epochs):
		print("epoch"*20, i)
		Delta = torch.randint(0, 180, size=(1,))

		env.reset(Delta=deg2rad(Delta), DT=1.7441998757264687e-14, DR=0.012178413663250922, Gamma=6)
		env.step(0., int(200/0.2))
		rewards = []
		angles = []
		prev_action = 0
		prev_reward = env.states[-1].O_R
		for j in range(measurements):

			print(f"Delta {int(rad2deg(env.states[-1].Delta.item()))}")
			action = ac(env.states[-1])
			print(f"action {actions_desc[action]:}")
			angles.append(env.states[-1].Delta.item())
			if use_means:
				env.step(actions[action], timestep_size, 1)
				reward = torch.tensor([s.O_R.item() for s in env.states[-timestep_size:]]).abs().mean().item()
			else:
				env.step(actions[action], 1, timestep_size)
				reward = abs(env.states[-1].O_R)

			print(f"O_R {env.states[-1].O_R:.2f}")
			bonus = 0
			if env.states[-1].Delta < 0:
				if actions[action] < 0:
					reward = 0.#-1 + 10 * (aps.Delta.item())
				elif actions[action] == 0:
					reward = 0.
				else:
					reward = .1
			elif env.states[-1].Delta > 3.1415926536:
				if actions[action] < 0:
					reward = .1
				elif actions[action] == 0:
					reward = 0.
				else:
					reward = 0.#-1 - 10 * (aps.Delta.item() - 3.1415926536)
			elif action == prev_action:
				if actions[action] == 0:
					bonus = 0.01
				else:
					bonus = 0.02
				print("Bonus: ^")
			elif actions[action] != 0 and actions[prev_action] != 0:
				bonus = -0.01
				print("Bonus: v")
			else:
				print("Bonus: -")

			if use_bonuses:
				if reward < 0:
					reward *= (1-bonus)
				else:
					reward *= (1+bonus)

			rel_reward = reward - prev_reward
			prev_reward = reward
			ac.rewards.append(rel_reward)
			print(f"reward {'+' if rel_reward >= 0.0 else ''}{rel_reward:.2f}")
			#print(f"reward {reward:.2f}")
			prev_action = action

		rewards += ac.rewards
		loss = ac.calculateLoss_old()
		print("loss", loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		ac.clear_memory()
		print("mean reward", torch.tensor(rewards).mean().item())
		print("mean angles", torch.tensor(angles).mean().item())

	weightpath = f"acdd_epochs{epochs}_measure{measurements}_timesteps{timestep_size}_{datetime.datetime.now():%Y%m%d-%H%M%S}.pt"
	print("Storing weights:", weightpath)
	torch.save(ac.state_dict(), weightpath)
