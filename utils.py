import torch
import torch.nn.functional as F

pi = 3.1415927410125732

deg2rad = lambda x: x * pi / 180.
rad2deg = lambda x: x * 180. / pi

def get_anglediff(a, b):
	"""Takes two complex vectors and calculates the (positive or negative)
	angle between them.
	"""
	aang = a.angle()
	bang = b.angle()

	diff = aang-bang
	diff = torch.where(diff <= -pi, diff % pi, diff) # if below -180deg convert it to positive equiv
	diff -= (diff >= pi) * pi * 2 # if above 180 deg remove 2 pi

	return diff


def angular_velocity(prev_state, cur_state):
	return get_anglediff(
		prev_state.positions-prev_state.positions.mean(),
		cur_state.positions-cur_state.positions.mean())


def torque(prev_state, cur_state):

	cm = cur_state.positions.mean()
	r = cur_state.positions-cm
	r /= r.abs()
	r = F.pad(torch.view_as_real(r), pad=(0,1,0,0), mode="constant", value=0)
	v = cur_state.positions - prev_state.positions
	v = F.pad(torch.view_as_real(v), pad=(0,1,0,0), mode="constant", value=0)
	e_z = torch.tensor([0,0,1], dtype=torch.float)

	return torch.mv(torch.cross(r, v), e_z)