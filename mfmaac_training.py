import os
import datetime
import gym
import torch
from utils import *
from MFMAAC import MFMAAC

# set the true number of threads
torch.set_num_threads(os.cpu_count())

env = gym.make('gym_swirl:swirl-v1')
env.seed(42)

if __name__ == "__main__":
	torch.set_printoptions(precision=2, sci_mode=False)
	torch.manual_seed(42)

	amount_particles = 48

	actions = deg2rad(torch.tensor([-1, 0, 1]))
	#actions_desc = torch.tensor(["<", "^", ">"])
	mfmaac = MFMAAC(num_inputs=1, num_actions=3)
	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#print("Model device", device)
	#mfmaac.to(device)

	optimizer = torch.optim.Adam(mfmaac.parameters(), lr=0.01, betas=(0.9, 0.999)) #lr=0.02
	epochs = 200
	measurements = 500
	timestep_size = 10 # 100*.2s*4deg/s = 80 deg (enough to see new behaviour)
	rewards = []
	angles = []
	use_means = False
	use_bonuses = True

	for epoch in range(epochs):
		print(f" EPOCH {epoch} ".center(80, "#"))
		Deltas = torch.randint(0, 180, size=(1,)).repeat(amount_particles)
		#Deltas.to(device)

		env.reset(Deltas=deg2rad(Deltas),
			DT=1.7441998757264687e-14,
			DR=0.012178413663250922,
			Gamma=6,
			amount=amount_particles)
		env.step(0., 1, int(200/0.2))
		rewards = []
		angles = []
		prev_action = 0
		target_OR = torch.tensor([1.0])
		prev_reward = env.states[-1].O_R
		for j in range(measurements):

			print(f"Deltas {rad2deg(env.states[-1].Deltas).mean().type(torch.int16)!r}")
			action = mfmaac(env.states[-1])
			#print(f"actions {action!r}")
			angles.append(env.states[-1].Deltas)

			env.step(actions[action], 1, timestep_size)
			ORs = env.states[-1].O_R.abs()
			#reward = 1 - (ORs - target_OR).abs()
			reward = ORs

			print(f"O_R {env.states[-1].O_R!r}")
			print(f"O_R mean {env.states[-1].O_R.mean()!r}")
			# bonus = 0
			negative_angle = (env.states[-1].Deltas < 0.0)
			small_reward = torch.tensor([.1])
			no_reward = torch.tensor([.0])
			reward = torch.where(negative_angle,
				torch.where(action == 2, small_reward, no_reward),
				reward)
			too_large_angle = (env.states[-1].Deltas > 3.1415926536)
			reward = torch.where(too_large_angle,
				torch.where(action == 0, small_reward, no_reward),
				reward)

			if use_bonuses:
				bonus = torch.where(action == prev_action, small_reward*.1, no_reward)
				reward *= torch.where(reward < 0., (1-bonus), (1+bonus))

			rel_reward = reward - prev_reward
			prev_reward = reward
			mfmaac.rewards.append(rel_reward)
			print(f"reward {rel_reward.mean()}")
			prev_action = action.clone()

		rewards += mfmaac.rewards
		loss = mfmaac.calculateLoss_old()
		print("loss", loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		mfmaac.clear_memory()

		states_path = os.path.join(os.path.dirname(__file__), "runs", f"mfmaac_train_epoch{epoch}")
		#env.save(states_path)
		#print("mean reward", torch.tensor(rewards).mean().item())
		#print("mean angles", torch.tensor(angles).mean().item())

	weightpath = f"mfmaac_epochs{epochs}_measure{measurements}_timesteps{timestep_size}_{datetime.datetime.now():%Y%m%d-%H%M%S}.pt"
	weightpath = os.path.join(os.path.dirname(__file__), "weights", weightpath)
	print("Storing weights:", weightpath)
	torch.save(mfmaac.state_dict(), weightpath)
