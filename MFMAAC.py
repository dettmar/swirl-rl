print("Start file")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
#from pytorch_lightning.core.lightning import LightningModule

print("Imported all")

class MFMAAC(nn.Module):

	def __init__(self, num_inputs=1, num_actions=10, hidden_size=128, learning_rate=3e-4):

		super(MFMAAC, self).__init__()

		self.affine = nn.Linear(num_inputs, hidden_size)
		self.action_layer = nn.Linear(hidden_size, num_actions)
		self.value_layer = nn.Linear(hidden_size, 1)

		self.action_layer.weight.data.fill_(1/hidden_size)
		self.action_layer.bias.data.fill_(1/hidden_size)

		self.logprobs = []
		self.state_values = []
		self.rewards = []


	def forward(self, state):
		Delta = state.Delta
		OR = state.O_R
		Delta = Delta.reshape((-1, 1)).float()
		Delta = F.relu(self.affine(Delta))
		#abs_O_R = torch.abs(O_R)
		state_value = self.value_layer(Delta)

		action_probs = F.softmax(self.action_layer(Delta))
		print("action_probs")
		print(f"{action_probs!r}")
		action_distribution = Categorical(action_probs)
		action = action_distribution.sample()
		self.logprobs.append(action_distribution.log_prob(action))

		#print("action_distribution.log_prob(action)", action_distribution.log_prob(action))
		self.state_values.append(state_value)

		return action


	def calculateLoss_old(self, gamma=0.99):

		# calculating discounted rewards:
		rewards = []
		dis_reward = 0
		for reward in self.rewards[::-1]:
			dis_reward = reward + gamma * dis_reward
			rewards.insert(0, dis_reward)

		#print("self.rewards", self.rewards)
		#print("rewards befo", rewards)

		# normalizing the rewards:

		rewards = torch.stack(rewards).float().squeeze()

		rewards = (rewards - rewards.mean(axis=0)) / (rewards.std(axis=0))
		#print("rewards afte", rewards)
		#rewards /= rewards.std()

		loss = torch.tensor(0.).reshape((1,))
		amount = self.rewards[0].numel()
		logprobs = torch.stack(self.logprobs).squeeze()
		values = torch.stack(self.state_values).squeeze()

		for i in range(amount):
			for logprob, value, reward in zip(logprobs.T[i], values.T[i], rewards.T[i]):
				advantage = reward - value.item()
				action_loss = -logprob * advantage
				value_loss = F.smooth_l1_loss(value, reward)
				loss += (action_loss + value_loss)

		return loss




	def clear_memory(self):
		del self.state_values[:]
		del self.logprobs[:]
		del self.rewards[:]


