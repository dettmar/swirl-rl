import os
import datetime
import gym
import torch
from tqdm import tqdm
from utils import *
from MFMAAC import MFMAAC

# set the true number of threads
torch.set_num_threads(os.cpu_count())

env = gym.make('gym_swirl:swirl-v1')
env.seed(10)

if __name__ == "__main__":
	torch.set_printoptions(precision=2, sci_mode=False)
	torch.manual_seed(10)

	amount_particles = 48

	actions = deg2rad(torch.tensor([-1, 0, 1]))
	#actions_desc = torch.tensor(["<", "^", ">"])
	mfmaac = MFMAAC(num_inputs=1, num_actions=3)
	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#tqdm.write("Model device", device)
	#mfmaac.to(device)


	optimizer = torch.optim.Adam(mfmaac.parameters(),
		lr=0.01,
		#weight_decay=0.9,
		betas=(0.9, 0.999)) #lr=0.02
	#optimizer = torch.optim.SGD(mfmaac.parameters(), lr=.01)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.99**x)
	epochs = 500
	measurements = 3000
	attempt = 38
	timestep_size = 10 # 10*.2s*4deg/s = max 8 deg (enough to see new behaviour)
	rewards = []
	angles = []
	use_means = False
	use_bonuses = True
	save_states = True
	save_intermediate_weights = True
	save_every_n = 50
	weight_dir = os.path.join(os.path.dirname(__file__), "weights")
	states_dir = os.path.join(os.path.dirname(__file__), "runs")
	epoch_log = tqdm(total=epochs, desc='Epochs', position=1)
	delta_log = tqdm(total=0, position=2, bar_format='{desc}')
	or_log = tqdm(total=0, position=3, bar_format='{desc}')
	reward_log = tqdm(total=0, position=4, bar_format='{desc}')
	action_log = tqdm(total=0, position=5, bar_format='{desc}')
	lr_log = tqdm(total=0, position=6, bar_format='{desc}')

	if any(f"mfmaac{attempt:03d}" in path for path in os.listdir(weight_dir)) or \
		any(f"mfmaac{attempt:03d}" in path for path in os.listdir(states_dir)):
		tqdm.write(f"Attempt {attempt:03d} already exists")
		exit()

	for epoch in range(epochs):
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
		timestep_log = tqdm(total=measurements*timestep_size, desc='Timestep', position=0)
		for j in range(measurements):
			
			delta_log.set_description_str(f"Deltas mean {rad2deg(env.states[-1].Deltas).mean().type(torch.int16).item()}, std {rad2deg(env.states[-1].Deltas).std().item():.2f}")
			action, action_probs = mfmaac(env.states[-1])
			action_log.set_description_str(f"Action probabilities {action_probs.mean(dim=0)!r}")
			#tqdm.write(f"actions {action!r}")
			angles.append(env.states[-1].Deltas)

			env.step(actions[action], 1, timestep_size)
			ORs = env.states[-1].O_R
			#reward = 1 - (ORs - target_OR).abs()
			reward = ORs#.abs() # using raw O_R for combatting opposing swirls

			#tqdm.write(f"O_R {env.states[-1].O_R!r}")
			or_log.set_description_str(f"O_R mean {'+' if env.states[-1].O_R.mean().item() >= 0.0 else ''}{env.states[-1].O_R.mean().item():.2f}, std {env.states[-1].O_R.std().item():.2f}")
			# bonus = 0
			negative_angle = (env.states[-1].Deltas < 0.0)
			small_reward = torch.tensor([.01])
			no_reward = torch.tensor([-1.])
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
			reward_log.set_description_str(f"Mean reward {'+' if rel_reward.mean() >= 0.0 else ''}{rel_reward.mean():.3f}")
			mfmaac.rewards.append(reward)#(rel_reward)
			prev_action = action.clone()

			if rad2deg(env.states[-1].Deltas).mean() > 200 or rad2deg(env.states[-1].Deltas).mean() < -20:
				tqdm.write("Particles too far outside range. Interrupting training.")
				break

			timestep_log.update(timestep_size)

		#tqdm.write("Measurements done")
		rewards += mfmaac.rewards
		#tqdm.write("rewq")
		loss = mfmaac.calculate_loss()
		#tqdm.write("loss", loss)

		for param_group in optimizer.param_groups:
			lr_log.set_description_str(f"Learning rate: {param_group['lr']}")
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		mfmaac.clear_memory()

		if save_states:
			states_path = os.path.join(states_dir, f"mfmaac{attempt:03d}_train_epoch{epoch}")
			env.save(states_path)
		#tqdm.write("mean reward", torch.tensor(rewards).mean().item())
		#tqdm.write("mean angles", torch.tensor(angles).mean().item())
		if save_intermediate_weights and epoch % save_every_n == 0 and epoch != 0:
			weightpath = f"mfmaac{attempt:03d}_epochs{epoch}of{epochs}_measure{measurements}_timesteps{timestep_size}_{datetime.datetime.now():%Y%m%d-%H%M%S}.pt"
			weightpath = os.path.join(weight_dir, weightpath)
			torch.save(mfmaac.state_dict(), weightpath)
			#tqdm.write("Storing weights:", weightpath)
		
		epoch_log.update(1)

	weightpath = f"mfmaac{attempt:03d}_epochs{epochs}_measure{measurements}_timesteps{timestep_size}_{datetime.datetime.now():%Y%m%d-%H%M%S}.pt"
	weightpath = os.path.join(os.path.dirname(__file__), "weights", weightpath)
	tqdm.write("Storing weights:", weightpath)
	torch.save(mfmaac.state_dict(), weightpath)
