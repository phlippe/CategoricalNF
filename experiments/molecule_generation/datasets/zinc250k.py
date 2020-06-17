import torch
import torch.utils.data as data
import numpy as np 
import pickle
import os
import sys
sys.path.append("../../../")

try:
	from rdkit import Chem
	ZINC250_BOND_DECODER = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
except:
	print("[!] WARNING: rdkit could not be imported. No evaluation and visualizations of molecules will be possible.")
	ZINC250_BOND_DECODER = {1: None, 2: None, 3: None}

from general.mutils import one_hot
from experiments.graph_coloring.datasets.mutils import BucketSampler
from experiments.molecule_generation.datasets.mutils import *


ZINC250_ATOMIC_NUM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]

class Zinc250kDataset(data.Dataset):

	DATASET_NODES = None
	DATASET_ADJENCIES = None
	DATASET_TRAIN_IDX = None
	DATASET_VAL_IDX = None
	DATASET_TRAIN_SMILES = None

	DATA_FILENAME = "zinc250k/zinc250k_compressed.npz"
	IDX_FILENAME = "zinc250k/zinc250k_dataidx.npz"
	LENGTH_PRIOR_FILENAME = "zinc250k/zinc250k_length_prior.npy"
	NODE_PRIOR_FILENAME = "zinc250k/zinc250k_node_prior.npy"
	EDGE_PRIOR_FILENAME = "zinc250k/zinc250k_edge_prior.npy"
	SMILES_FILENAME = "zinc250k/zinc250k_train_smiles.pik"


	def __init__(self, train=False, val=False, test=False, data_root="data/"):
		super().__init__()
		self.train, self.val, self.test = train, val, test
		Zinc250kDataset.load_dataset(data_root=data_root)
		if self.train:
			self.data_indices = Zinc250kDataset.DATASET_TRAIN_IDX
		elif self.val:
			self.data_indices = Zinc250kDataset.DATASET_VAL_IDX[:8192]
		else:
			self.data_indices = Zinc250kDataset.DATASET_VAL_IDX[8192:]
		print("Num %s examples: %i" % ("training" if self.train else ("validation" if self.val else "testing"), self.data_indices.shape[0]))


	def __len__(self):
		return self.data_indices.shape[0]


	def __getitem__(self, idx):
		idx = self.data_indices[idx]
		nodes = Zinc250kDataset.DATASET_NODES[idx].astype(np.int64)
		adjacency = Zinc250kDataset.DATASET_ADJENCIES[idx].astype(np.int64)
		length = (nodes >= 0).sum().astype(np.int64)
		nodes = nodes + (nodes == -1) # Setting padding to 0
		return nodes, adjacency, length


	@staticmethod
	def load_dataset(data_root="../data/"):
		if any([e is None for e in [Zinc250kDataset.DATASET_NODES, 
									Zinc250kDataset.DATASET_ADJENCIES]]):
			print("Loading Zinc250k dataset...")
			data_arr = np.load(os.path.join(data_root, Zinc250kDataset.DATA_FILENAME))
			nodes, adjacency = data_arr["nodes"], data_arr["adjacency"]
			Zinc250kDataset.DATASET_NODES = nodes
			Zinc250kDataset.DATASET_ADJENCIES = adjacency
			print("Dataset loaded")

		if Zinc250kDataset.DATASET_VAL_IDX is None:
			data_arr = np.load(os.path.join(data_root, Zinc250kDataset.IDX_FILENAME))
			Zinc250kDataset.DATASET_TRAIN_IDX = data_arr["train_idx"]
			Zinc250kDataset.DATASET_VAL_IDX = data_arr["val_idx"]

		if Zinc250kDataset.DATASET_TRAIN_SMILES is None:
			smiles_file = os.path.join(data_root, Zinc250kDataset.SMILES_FILENAME)
			if os.path.isfile(smiles_file):
				with open(smiles_file, "rb") as f:
					train_smiles = pickle.load(f)
				Zinc250kDataset.DATASET_TRAIN_SMILES = train_smiles
			else:
				print("[!] WARNING: Train smiles file missing.")


	@staticmethod
	def get_length_prior(data_root="data/"):
		file_path = os.path.join(data_root, Zinc250kDataset.LENGTH_PRIOR_FILENAME)
		if not os.path.isfile(file_path):
			Zinc250kDataset.load_dataset(data_root=data_root)
			log_length_prior = calculate_length_prior(Zinc250kDataset.DATASET_NODES)
			np.save(file_path, log_length_prior)
		log_length_prior = np.load(file_path)
		return log_length_prior


	@staticmethod
	def graph_to_mol(nodes, adjacency, allow_submolecule=True):
		global ZINC250_ATOMIC_NUM_LIST, ZINC250_BOND_DECODER

		if isinstance(nodes, torch.Tensor):
			nodes = nodes.detach().cpu().numpy()
		if isinstance(adjacency, torch.Tensor):
			adjacency = adjacency.detach().cpu().numpy()
		if allow_submolecule:
			nodes, adjacency = find_largest_submolecule(nodes, adjacency)

		mol = Chem.RWMol()
		for n in nodes:
			if n < 0:
				continue
			mol.AddAtom(Chem.Atom(int(ZINC250_ATOMIC_NUM_LIST[n])))

		for i in range(adjacency.shape[0]):
			for j in range(i+1, adjacency.shape[1]):
				if adjacency[i,j] == 0:
					continue
				mol.AddBond(i, j, ZINC250_BOND_DECODER[adjacency[i,j]])
		return mol

	@staticmethod
	def visualize_molecule(nodes, adjacency, filename="test_img"):
		mol = Zinc250kDataset.graph_to_mol(nodes, adjacency)
		visualize_molecule(mol, filename=filename)

	@staticmethod
	def evaluate_generations(*args, **kwargs):
		return evaluate_generations(Zinc250kDataset, *args, **kwargs)

	@staticmethod
	def _check_validity(mol):
		s = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) if mol is not None else None
		if s is not None and "." not in Chem.MolToSmiles(s):
			return s 
		else:
			return None

	@staticmethod
	def _check_novelty(mol, smiles=None):
		if smiles is None:
			smiles = Chem.MolToSmiles(mol)
		return smiles not in Zinc250kDataset.DATASET_TRAIN_SMILES

	@staticmethod
	def num_edge_types():
		return 3

	@staticmethod
	def num_node_types():
		global ZINC250_ATOMIC_NUM_LIST
		return len(ZINC250_ATOMIC_NUM_LIST)

	@staticmethod
	def num_max_neighbours():
		return 5

	@staticmethod
	def max_num_nodes():
		return 38

	@staticmethod
	def get_node_prior(data_root="data/"):
		Zinc250kDataset.load_dataset(data_root=data_root)
		file_path = os.path.join(data_root, Zinc250kDataset.NODE_PRIOR_FILENAME)
		if not os.path.isfile(file_path):
			np.save(file_path, calculate_node_distribution(Zinc250kDataset))
		return np.load(file_path)

	@staticmethod
	def get_edge_prior(data_root="data/"):
		Zinc250kDataset.load_dataset(data_root=data_root)
		file_path = os.path.join(data_root, Zinc250kDataset.EDGE_PRIOR_FILENAME)
		if not os.path.isfile(file_path):
			np.save(file_path, calculate_edge_distribution(Zinc250kDataset))
		return np.load(file_path)

	def get_sampler(self, batch_size, drop_last=False, **kwargs):
		return data.BatchSampler(BucketSampler(self, batch_size, len_step=3), batch_size, drop_last=drop_last)

	


if __name__ == '__main__':
	Zinc250kDataset.load_dataset(data_root="../data/")
	
	#########################
	## Plotting statistics ##
	#########################
	plot_dataset_statistics(Zinc250kDataset, show_plots=True)

	##################################
	## Visualizing random molecules ##
	##################################
	# dataset = Zinc250kDataset(train=True, val=False, test=False, data_root="../data/")
	# print("Training dataset length", len(dataset))
	# dataset_val = Zinc250kDataset(train=False, val=True, test=False, data_root="../data/")
	# print("Validation dataset length", len(dataset_val))
	# for i in range(4):
	# 	nodes, adjacency, length = dataset[i]
	# 	Zinc250kDataset.visualize_molecule(nodes, adjacency, filename="example_molecule_%i" % (i+1))

	#############################
	## Testing valid molecules ##
	#############################
	# checkpoint_path = ... # Add the checkpoint to analyse here!
	# a = np.load(os.path.join(checkpoint_path, "gens_molecules.npz"))
	# nodes, adjacency, length = a["z"], a["adjacency"], a["length"]
	# valid_list = analyse_generations(nodes, adjacency, length)
	
	# def visu_idx(idx):
	# 	Zinc250kDataset.visualize_molecule(nodes=nodes[idx,:length[idx]], adjacency=adjacency[idx,:length[idx],:length[idx]], 
	# 									   filename=os.path.join(checkpoint_path, "generated_molecule_%i_%s" % (idx, "v" if valid_list[idx]==1 else "nv")))

	# num_visus = 0
	# for i in range(10):
	#	visu_idx(i)