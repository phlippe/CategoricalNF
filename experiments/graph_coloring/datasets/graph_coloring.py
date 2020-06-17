import torch
import torch.utils.data as data
import numpy as np 
import random
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import networkx as nx
import os
import sys
sys.path.append("../../../")

from experiments.graph_coloring.datasets.mutils import BucketSampler

NUM_COLORS = 3
PREFIX = "_large" # or "_tiny"


class GraphColoringDataset(data.Dataset):

	DATASET_NODES = None
	DATASET_ADJACENCIES = None
	DATASET_TRAIN_IDX = None
	DATASET_VAL_IDX = None
	DATASET_TEST_IDX = None
	DATA_FILENAME = "graph_coloring_compressed_%i%s.npz" % (NUM_COLORS, PREFIX)
	IDX_FILENAME = "graph_coloring_dataidx_%i%s.npz" % (NUM_COLORS, PREFIX)

	def __init__(self, num_colors=NUM_COLORS, train=False, val=False, test=False, order_graphs="none", data_root="data/"):
		super().__init__()
		self.train, self.val, self.test = train, val, test
		self.num_colors = num_colors
		GraphColoringDataset.load_dataset(data_root=data_root)
		if self.train:
			self.data_indices = GraphColoringDataset.DATASET_TRAIN_IDX 
		elif self.val:
			self.data_indices = GraphColoringDataset.DATASET_VAL_IDX
		else:
			self.data_indices = GraphColoringDataset.DATASET_TEST_IDX
		self.order_graphs = order_graphs
		assert self.order_graphs in ["none", "rand", "largest_first", "smallest_first"], "[!] ERROR: Order \"%s\" unknown" % self.order_graphs
		print("Num %s examples: %i" % ("training" if self.train else ("validation" if self.val else "testing"), self.data_indices.shape[0]))


	def __len__(self):
		return self.data_indices.shape[0]


	def __getitem__(self, idx):
		idx = self.data_indices[idx]
		nodes = GraphColoringDataset.DATASET_NODES[idx].astype(np.int64)
		adjacency = GraphColoringDataset.DATASET_ADJACENCIES[idx].astype(np.int64)
		length = (nodes >= 0).sum().astype(np.int64)
		if self.train:
			nodes[:length] = self._shuffle_colors(nodes[:length])
		nodes = nodes + (nodes == -1) # Setting padding to 0
		adjacency = adjacency + (adjacency == -1)
		nodes, adjacency = self._sort_nodes(nodes, adjacency, length)
		return nodes, adjacency, length

	def _shuffle_colors(self, nodes):
		colors = [i for i in range(self.num_colors)]
		random.shuffle(colors)
		nodes = np.array(colors)[nodes]
		return nodes

	def _sort_nodes(self, nodes, adjacency, length):
		if self.order_graphs == "none":
			return nodes, adjacency 
		elif self.order_graphs == "rand":
			pos = [i for i in range(length)]
			random.shuffle(pos)
			pos = pos + [i for i in range(length, nodes.shape[0])]
			pos = np.array(pos)
		elif self.order_graphs == "largest_first":
			num_neighbours = (adjacency > 0).astype(np.float32).sum(axis=1)
			num_neighbours = num_neighbours + np.random.uniform(size=num_neighbours.shape) * 1e-2
			pos = np.argsort(num_neighbours)[::-1]
		elif self.order_graphs == "smallest_first":
			num_neighbours = (adjacency > 0).astype(np.float32).sum(axis=1)
			num_neighbours = num_neighbours + (num_neighbours == 0) * 100
			num_neighbours = num_neighbours + np.random.uniform(size=num_neighbours.shape) * 1e-2
			pos = np.argsort(num_neighbours)
		else:
			return nodes, adjacency
		
		nodes = nodes[pos]
		adjacency = adjacency[pos,:][:,pos]
		return nodes, adjacency


	@staticmethod
	def load_dataset(data_root="data/"):
		global PREFIX, NUM_COLORS

		if any([e is None for e in [GraphColoringDataset.DATASET_NODES, 
									GraphColoringDataset.DATASET_ADJACENCIES]]):
			print("Loading graph coloring dataset (prefix=%s, %i colors)..." % (PREFIX, NUM_COLORS))
			filepath = os.path.join(data_root, GraphColoringDataset.DATA_FILENAME)
			assert os.path.isfile(filepath), "[!] ERROR: The graph coloring dataset could not be loaded due to a missing file.\n" + \
											 "Make sure that the data is placed at: \"%s\"" % str(filepath)
			data_arr = np.load(filepath)
			nodes, adjacency = data_arr["nodes"], data_arr["adjacency"]
			GraphColoringDataset.DATASET_NODES = nodes
			GraphColoringDataset.DATASET_ADJACENCIES = adjacency
			print("Dataset loaded")

		if GraphColoringDataset.DATASET_VAL_IDX is None:
			data_arr = np.load(os.path.join(data_root, GraphColoringDataset.IDX_FILENAME))
			GraphColoringDataset.DATASET_TRAIN_IDX = data_arr["train_idx"]
			GraphColoringDataset.DATASET_VAL_IDX = data_arr["val_idx"]
			GraphColoringDataset.DATASET_TEST_IDX = data_arr["test_idx"]


	@staticmethod
	def evaluate_generations(nodes, adjacency, length=None, **kwargs):
		if isinstance(nodes, torch.Tensor):
			nodes = nodes.detach().cpu().numpy()
		if isinstance(adjacency, torch.Tensor):
			adjacency = adjacency.detach().cpu().numpy()
		if length is None:
			length = np.zeros(nodes.shape[0]) + nodes.shape[1]
		elif isinstance(length, torch.Tensor):
			length = length.detach().cpu().numpy()

		valid = [1 if GraphColoringDataset._check_validity(nodes=nodes[i], adjacency=adjacency[i], length=length[i]) else 0 for i in range(nodes.shape[0])]
		valid_ratio = sum(valid) * 1.0 / len(valid)

		eval_dict = {
			"valid_ratio": valid_ratio
		}

		return eval_dict


	@staticmethod
	def _check_validity(nodes, adjacency, length=None):
		if length is not None:
			nodes = nodes[:length]
			adjacency = adjacency[:length,:length]
		nodes = nodes + 1 # To numerate colors from 1 to num_colors
		labeled_adjacency = adjacency * nodes[None,:]
		return np.all(nodes[:,None] != labeled_adjacency)


	@staticmethod
	def num_node_types():
		return NUM_COLORS


	@staticmethod
	def set_dataset(prefix="_tiny", num_colors=3):
		global PREFIX, NUM_COLORS
		PREFIX = prefix
		NUM_COLORS = num_colors
		GraphColoringDataset.DATA_FILENAME = "graph_coloring_compressed_%i%s.npz" % (NUM_COLORS, PREFIX)
		GraphColoringDataset.IDX_FILENAME = "graph_coloring_dataidx_%i%s.npz" % (NUM_COLORS, PREFIX)


	def get_sampler(self, batch_size, drop_last=False, **kwargs):
		return data.BatchSampler(BucketSampler(self, batch_size, len_step=1), batch_size, drop_last=drop_last)


def plot_by_adjacency_matrix(adjacency, nodes=None, show_plot=True, **kwargs):
	G = nx.convert_matrix.from_numpy_array(adjacency)
	if nodes is not None and "node_color" not in kwargs:
		unique_nodes = np.unique(nodes)
		colors = [hsv_to_rgb(np.array([c*1.0/unique_nodes.shape[0], 1.0, 1.0])) for c in range(unique_nodes.shape[0])]
		kwargs["node_color"] = [colors[np.where(unique_nodes==nodes[i])[0][0]] for i in range(nodes.shape[0])]
	nx.draw(G, **kwargs)
	if show_plot:
		plt.show()


if __name__ == '__main__':
	torch.manual_seed(42)
	np.random.seed(42)
	random.seed(42)

	GraphColoringDataset.set_dataset(prefix="_tiny")
	GraphColoringDataset.load_dataset(data_root="../data/")

	#########################
	## Visualizing a graph ##
	#########################
	# dataset = GraphColoringDataset(train=True, val=False, test=False, data_root="../data/")
	# for i in range(10):
	# 	nodes, adjacency, length = dataset[i]
	# 	nodes = nodes[:length]
	# 	adjacency = adjacency[:length,:length]
	# 	print("Number of nodes", length)
	# 	plot_by_adjacency_matrix(adjacency, nodes=nodes, node_size=400, show_plot=True)
	# 	plt.close()

	arr = np.load("../checkpoints/large_3_CNF/node_samples.npz")
	nodes = arr["z"][:,:,0]
	adjacency = arr["adjacency"]
	length = arr["length"]
	print(GraphColoringDataset.evaluate_generations(nodes, adjacency, length))
