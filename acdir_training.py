import os
import datetime
import gym
import torch
from ACDir import ActorCriticDiscrete
from utils import *

torch.set_num_threads(24)#os.cpu_count()
seed = 40
env = gym.make('gym_swirl:swirl-v1')
env.seed(seed)
torch.manual_seed(seed)


if __name__ == "__main__":

	amount_particles = 48
	actions = deg2rad(torch.tensor([-1, 1]))
	actions_desc = ["<", ">"]
	ac = ActorCriticDiscrete(num_inputs=1, num_actions=2)

	optimizer = torch.optim.Adam(ac.parameters(), lr=0.01, betas=(0.9, 0.999)) #lr=0.02
	attempt = 12
	epochs = 500
	measurements = 2000
	timestep_size = 10 # 100*.2s*4deg/s = 80 deg (enough to see new behaviour)
	rewards = []
	angles = []
	small_reward = torch.tensor([.00])
	no_reward = torch.tensor([-0.1])
	use_bonuses = True
	save_states = True
	save_intermediate_weights = True
	keep_latest = True
	save_every_n = 20
	weight_dir = os.path.join(os.path.dirname(__file__), "weights")
	states_dir = os.path.join(os.path.dirname(__file__), "runs")
	lr_decay_start = 50
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 0.95**(ep-lr_decay_start) if ep > lr_decay_start else 1.0)

	if any(path.startswith(f"acdd{attempt:03d}") for path in os.listdir(weight_dir)) or \
		any(path.startswith(f"acdd{attempt:03d}") for path in os.listdir(states_dir)):
		print(f"Attempt {attempt:03d} already exists")
		exit()

	for epoch in range(epochs):
		print("epoch"*20, epoch)
		Delta = torch.randint(0, 180, size=(1,))

		env.reset(Deltas=deg2rad(Delta),
			DT=1.7441998757264687e-14,
			DR=0.012178413663250922,
			Gamma=6,
			amount=amount_particles)
		env.step(0., int(200/0.2))
		rewards = []
		angles = []
		prev_action = 0
		prev_reward = env.states[-1].O_R.mean()
		reached_negative_OR = False
		for m in range(measurements):

			print(f"Delta {int(rad2deg(env.states[-1].Deltas.item()))}")
			action = ac(env.states[-1])
			print(f"action {actions_desc[action]:}")
			angles.append(env.states[-1].Deltas.item())

			env.step(actions[action], 1, timestep_size)
			reward = env.states[-1].O_R.mean()

			negative_angle = (env.states[-1].Deltas.mean() < 0.0)
			if negative_angle:
				reward = no_reward #small_reward if action == 1 else 
			elif env.states[-1].Deltas.mean() > 3.1415926536:
				reward = no_reward #small_reward if action == 0 else 

			rel_reward = reward - prev_reward
			prev_reward = reward
			ac.rewards.append(rel_reward)
			print(f"reward {'+' if rel_reward.item() >= 0.0 else ''}{rel_reward.item():.2f}")
			#print(f"reward {reward:.2f}")
			prev_action = action

			if rad2deg(env.states[-1].Deltas).mean() > 200 or rad2deg(env.states[-1].Deltas).mean() < -20:
				print("Delta too far outside range. Interrupting training.")
				break

			if reward < -.5:
				reached_negative_OR = True
				print("OR negative. Interrupting training.")
				break
		
		if keep_latest:
			weightpath = f"acdd{attempt:03d}_epochs{epochs}_latest.pt"
			weightpath = os.path.join(weight_dir, weightpath)
			torch.save(ac.state_dict(), weightpath)

		if reached_negative_OR:
			print("Not learning")
			ac.clear_memory()
			reached_negative_OR = False
			continue

		if save_states:
			states_path = os.path.join(states_dir, f"acdd{attempt:03d}_train_epoch{epoch}")
			env.save(states_path)

		rewards += ac.rewards
		loss = ac.calculateLoss()
		print("loss", loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		ac.clear_memory()
		print("mean reward", torch.tensor(rewards).mean().item())
		print("mean angles", torch.tensor(angles).mean().item())
		print("save_intermed", save_intermediate_weights and epoch % save_every_n == 0 and epoch > 0, epoch % save_every_n, epoch)
		if save_intermediate_weights and epoch % save_every_n == 0 and epoch > 0:
			weightpath = f"acdd{attempt:03d}_epochs{epoch}of{epochs}_measure{measurements}_timesteps{timestep_size}_{datetime.datetime.now():%Y%m%d-%H%M%S}.pt"
			weightpath = os.path.join(weight_dir, weightpath)
			print("Storing weights:", weightpath)
			torch.save(ac.state_dict(), weightpath)

	weightpath = f"acdd_epochs{epochs}_measure{measurements}_timesteps{timestep_size}_{datetime.datetime.now():%Y%m%d-%H%M%S}.pt"
	weightpath = os.path.join(weight_dir, weightpath)
	print("Storing weights:", weightpath)
	torch.save(ac.state_dict(), weightpath)
