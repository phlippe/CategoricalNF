import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
np.random.seed(43) 
plt.figure(figsize=(6,6))
sns.set()



use_simple_mixtures = False
show_partitioning = False

fig_name = "encoding_dist_" + ("mixture" if use_simple_mixtures else "linear_flows")
if not show_partitioning:
	fig_name += "_pure"

green_shades = sns.cubehelix_palette(start=5.0/3.0, rot=0.0, hue=2.0, gamma=1.0, dark=0.2, as_cmap=True)
blue_shades = sns.cubehelix_palette(start=8.0/3.0, rot=0.0, hue=2.0, gamma=1.0, dark=0.2, as_cmap=True)
red_shades = sns.cubehelix_palette(start=1.0, rot=0.0, hue=1.0, gamma=1.0, dark=0.2, as_cmap=True)


def sort_to_line(vals):
	"""
	Function to sort border points to a line
	"""
	point_list = [vals[0]]
	vals = np.delete(vals, 0, axis=0)
	while len(vals) > 0:
		dists = ((point_list[-1]-vals)**2).sum(axis=-1)
		idx = np.argmin(dists)
		if vals[idx,0]!=point_list[-1][0] and vals[idx,1]!=point_list[-1][1]:
			point_list.append(vals[idx])
		vals = np.delete(vals, idx, axis=0)
	point_list = np.array(point_list)
	return point_list


def logdist(x, mu=0, sigma=1.0):
	"""
	Returns the probability value of a logistic distribution with given mean and stdev for input points x
	"""
	x = (x - mu) / sigma
	return np.exp(-x) / (1 + np.exp(-x))**2

res = 1024
xx, yy = np.meshgrid(np.linspace(-5, 7.5, res), np.linspace(-5, 7.5, res))

if use_simple_mixtures:
	mixtures = [([-2, 0], [0.6, 0.4], green_shades),
				([5, -2.5], [0.3, 0.5], blue_shades),
				([4.5, 5], [0.5, 0.5], red_shades)
			   ]
	mixt_probs = []
	for mu, sigma, color in mixtures:
		prob = logdist(xx, mu[0], sigma[0]) * logdist(yy, mu[1], sigma[1])
		mixt_probs.append(prob)
		plt.contourf(np.ma.masked_less(prob, np.max(prob)/5), cmap=color, zorder=5)
else:
	mixtures = [([-2, 2], 0, green_shades),
				([5, -2.5], 1, blue_shades),
				([4.5, 5], 2, red_shades)
			   ]
	mixt_probs = []
	for mu, sigma, color in mixtures:
		if sigma == 0:
			yy1 = yy
			xx1 = xx + 0.4*np.abs(yy - mu[1])**2
			prob = logdist(xx1, mu[0], 0.3) * logdist(yy1, mu[1], 0.8)
		elif sigma == 1:
			prob  = 0.7  * logdist(xx, mu[0], 0.3) * logdist(yy, mu[1], 0.5)
			prob += 0.3  * logdist(xx, mu[0]-2, 0.5) * logdist(yy, mu[1], 0.5)
		elif sigma == 2:
			xy = np.stack([xx, yy], axis=-1)
			angle = -45/360*2*np.pi
			rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
			rot_xy = (xy[:,:,:,None] * rot_matrix[None,None,:,:]).sum(axis=2)
			mu1 = mu[0] * rot_matrix[0,0] + mu[1] * rot_matrix[1,0]
			mu2 = mu[0] * rot_matrix[0,1] + mu[1] * rot_matrix[1,1]
			prob = logdist(rot_xy[:,:,0], mu1, 0.6) * logdist(rot_xy[:,:,1], mu2, 0.3)
		
		mixt_probs.append(prob)
		plt.contourf(np.ma.masked_less(prob, np.max(prob)/5), cmap=color, zorder=5)

if show_partitioning:
	mixt_probs = np.stack(mixt_probs, axis=-1)
	mixt_probs = mixt_probs / mixt_probs.max(axis=-1, keepdims=True)
	softmax = mixt_probs / mixt_probs.sum(axis=-1, keepdims=True)
	colors = np.array([green_shades(0.0), blue_shades(0.0), red_shades(0.0)])
	partition = (softmax[:,:,:,None] * colors[None,None,:,:]).sum(axis=-2)
	partition = np.minimum(partition, 1.0)
	partition[:,:,3] = 0.8
	plt.imshow(partition, origin='upper', zorder=1)

	colors = np.array([green_shades(1.0), blue_shades(1.0), red_shades(1.0)])
	softmax_diff_y = - (softmax[:,1:] - 0.95) * (softmax[:,:-1] - 0.95)
	softmax_diff_x = - (softmax[1:,:] - 0.9) * (softmax[:-1,:] - 0.9)
	for i, (_, _, color) in enumerate(mixtures):
		idx = np.where(softmax_diff_y[:-1,:,i]>0)
		idx_list = np.stack([idx[0], idx[1]], axis=-1)
		idx = sort_to_line(idx_list)
		plt.plot(idx[:,1], idx[:,0], '-.', color=color(0.5), linewidth=2.5)
else:
	plt.gca().invert_yaxis()

frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.tight_layout()
plt.savefig(fig_name + ".png", dpi=150)