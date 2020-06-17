import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("../../../")

try:
	from rdkit import Chem
	MOSES_BOND_DECODER = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
except:
	print("[!] WARNING: rdkit could not be imported. No evaluation and visualizations of molecules will be possible.")
	MOSES_BOND_DECODER = {1: None, 2: None, 3: None}

from general.mutils import one_hot
from experiments.graph_coloring.datasets.mutils import BucketSampler
from experiments.molecule_generation.datasets.mutils import *


MOSES_ATOMIC_NUM_LIST = [6, 7, 8, 9, 16, 17, 35, 0]  # 0 is for virtual node. 

class MosesDataset(data.Dataset):

	DATASET_NODES = None
	DATASET_ADJENCIES = None
	DATASET_TRAIN_IDX = None
	DATASET_VAL_IDX = None
	DATASET_TRAIN_SMILES = None
	DATA_FILENAME = "moses/moses_compressed.npz"
	IDX_FILENAME = "moses/moses_dataidx.npz"
	LENGTH_PRIOR_FILENAME = "moses/moses_length_prior.npy"
	NODE_PRIOR_FILENAME = "moses/moses_node_prior.npy"
	EDGE_PRIOR_FILENAME = "moses/moses_edge_prior.npy"
	SMILES_FILENAME = "moses/train.csv"


	def __init__(self, train=False, val=False, test=False, data_root="data/"):
		super().__init__()
		self.train, self.val, self.test = train, val, test
		MosesDataset.load_dataset(data_root=data_root)
		if self.train:
			self.data_indices = MosesDataset.DATASET_TRAIN_IDX
		elif self.val:
			self.data_indices = MosesDataset.DATASET_VAL_IDX[:8192]
		else:
			self.data_indices = MosesDataset.DATASET_VAL_IDX[8192:]
		print("Num %s examples: %i" % ("training" if self.train else ("validation" if self.val else "testing"), self.data_indices.shape[0]))


	def __len__(self):
		return self.data_indices.shape[0]


	def __getitem__(self, idx):
		idx = self.data_indices[idx]
		nodes = MosesDataset.DATASET_NODES[idx].astype(np.int64)
		adjacency = MosesDataset.DATASET_ADJENCIES[idx].astype(np.int64)
		length = (nodes >= 0).sum().astype(np.int64)
		nodes = nodes + (nodes == -1) # Setting padding to 0
		return nodes, adjacency, length


	@staticmethod
	def load_dataset(data_root="../data/"):
		if any([e is None for e in [MosesDataset.DATASET_NODES, 
									MosesDataset.DATASET_ADJENCIES]]):
			print("Loading Moses dataset...")
			data_arr = np.load(os.path.join(data_root, MosesDataset.DATA_FILENAME))
			nodes, adjacency = data_arr["nodes"], data_arr["adjacency"]
			MosesDataset.DATASET_NODES = nodes
			MosesDataset.DATASET_ADJENCIES = adjacency
			print("Dataset loaded")

		if os.path.isfile(os.path.join(data_root, MosesDataset.IDX_FILENAME)) and MosesDataset.DATASET_VAL_IDX is None:
			data_arr = np.load(os.path.join(data_root, MosesDataset.IDX_FILENAME))
			MosesDataset.DATASET_TRAIN_IDX = data_arr["train_idx"]
			MosesDataset.DATASET_VAL_IDX = data_arr["val_idx"]

		if os.path.isfile(os.path.join(data_root, MosesDataset.SMILES_FILENAME)) and MosesDataset.DATASET_TRAIN_SMILES is None:
			smiles_file = os.path.join(data_root, MosesDataset.SMILES_FILENAME)
			if os.path.isfile(smiles_file):
				MosesDataset.DATASET_TRAIN_SMILES = set(
														pd.read_csv(os.path.join(data_root, "moses/train.csv"),
																	 usecols=['SMILES'],
																	 squeeze=True).astype(str).tolist()
														)
			else:
				print("[!] WARNING: Train smiles file missing.")


	@staticmethod
	def get_length_prior(data_root="data/"):
		file_path = os.path.join(data_root, MosesDataset.LENGTH_PRIOR_FILENAME)
		if not os.path.isfile(file_path):
			MosesDataset.load_dataset(data_root=data_root)
			log_length_prior = calculate_length_prior(MosesDataset.DATASET_NODES)
			np.save(file_path, log_length_prior)
		log_length_prior = np.load(file_path)
		return log_length_prior


	@staticmethod
	def graph_to_mol(nodes, adjacency, allow_submolecule=False):
		global MOSES_ATOMIC_NUM_LIST, MOSES_BOND_DECODER

		if isinstance(nodes, torch.Tensor):
			nodes = nodes.detach().cpu().numpy()
		if isinstance(adjacency, torch.Tensor):
			adjacency = adjacency.detach().cpu().numpy()
		
		invalid_atoms = (nodes < 0) | nodes >= MosesDataset.num_node_types()
		if np.any(invalid_atoms) and not np.all(invalid_atoms):
			valid_idx = np.where(~invalid_atoms)[0]
			nodes = nodes[valid_idx]
			adjacency = adjacency[valid_idx,:][:,valid_idx]

		if allow_submolecule:
			nodes, adjacency = find_largest_submolecule(nodes, adjacency)

		mol = Chem.RWMol()
		for n in nodes:
			if n < 0:
				continue
			mol.AddAtom(Chem.Atom(int(MOSES_ATOMIC_NUM_LIST[n])))

		for i in range(adjacency.shape[0]):
			for j in range(i+1, adjacency.shape[1]):
				if adjacency[i,j] == 0:
					continue
				mol.AddBond(i, j, MOSES_BOND_DECODER[adjacency[i,j]])
		return mol

	@staticmethod
	def visualize_molecule(nodes, adjacency, filename="test_img"):
		mol = MosesDataset.graph_to_mol(nodes, adjacency)
		visualize_molecule(mol, filename=filename)

	@staticmethod
	def evaluate_generations(*args, **kwargs):
		return evaluate_generations(MosesDataset, *args, **kwargs)

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
		return smiles not in MosesDataset.DATASET_TRAIN_SMILES

	@staticmethod
	def num_edge_types():
		return 3

	@staticmethod
	def num_node_types():
		global MOSES_ATOMIC_NUM_LIST
		return len(MOSES_ATOMIC_NUM_LIST)-1

	@staticmethod
	def num_max_neighbours():
		return 4

	@staticmethod
	def max_num_nodes():
		return 27

	@staticmethod
	def get_node_prior(data_root="data/"):
		MosesDataset.load_dataset(data_root=data_root)
		file_path = os.path.join(data_root, MosesDataset.NODE_PRIOR_FILENAME)
		if not os.path.isfile(file_path):
			np.save(file_path, calculate_node_distribution(MosesDataset))
		return np.load(file_path)

	@staticmethod
	def get_edge_prior(data_root="data/"):
		MosesDataset.load_dataset(data_root=data_root)
		file_path = os.path.join(data_root, MosesDataset.EDGE_PRIOR_FILENAME)
		if not os.path.isfile(file_path):
			np.save(file_path, calculate_edge_distribution(MosesDataset))
		return np.load(file_path)

	def get_sampler(self, batch_size, drop_last=False, **kwargs):
		return data.BatchSampler(BucketSampler(self, batch_size, len_step=2), batch_size, drop_last=drop_last)



if __name__ == '__main__':
	MosesDataset.load_dataset(data_root="../data/")
	MosesDataset(train=True, data_root="../data/")
	MosesDataset(val=True, data_root="../data/")
	MosesDataset(test=True, data_root="../data/")

	#########################
	## Plotting statistics ##
	#########################
	plot_dataset_statistics(MosesDataset, show_plots=True)

	#############################
	## Testing valid molecules ##
	#############################
	# checkpoint_path = ... # Add the checkpoint to analyse here!
	# a = np.load(os.path.join(checkpoint_path, "gens_molecules.npz"))
	# nodes, adjacency, length = a["z"], a["adjacency"], a["length"]
	# valid_list = analyse_generations(nodes, adjacency, length)

	# def visu_idx(idx):
	# 	print("Nodes", nodes[idx,:length[idx]])
	# 	MosesDataset.visualize_molecule(nodes=nodes[idx,:length[idx]], adjacency=adjacency[idx,:length[idx],:length[idx]], 
	# 									   filename=os.path.join(checkpoint_path, "generated_molecule_%i%s" % (idx, "v" if valid_list[idx]==1 else "nv")))

	# for i in range(10):
	# 	visu_idx(i)