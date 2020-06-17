import torch
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
try:
	from rdkit import Chem
	from rdkit.Chem import Draw
	from rdkit.Chem import AllChem
	from rdkit import RDLogger
	lg = RDLogger.logger()
	lg.setLevel(RDLogger.CRITICAL)
	ZINC250_BOND_DECODER = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
	RDKIT_IMPORTED = True
except:
	print("[!] WARNING: rdkit could not be imported. No evaluation and visualizations of molecules will be possible.")
	RDKIT_IMPORTED = False
try:
	import cairosvg
	SVG2PDF_IMPORTED = True
except:
	print("[!] WARNING: cairosvg could not be imported. Visualizations of molecules cannot be converted to pdf.")
	SVG2PDF_IMPORTED = False



def plot_dataset_statistics(dataset_class, data_root="../data/", show_plots=True):
	dataset_class.load_dataset(data_root=data_root)
	log_length_prior = dataset_class.get_length_prior(data_root=data_root)
	length_distribution = (dataset_class.DATASET_NODES >= 0).sum(axis=1).astype(np.int32)
	node_distribution = dataset_class.DATASET_NODES[np.where(dataset_class.DATASET_NODES >= 0)]
	edge_distribution = [((dataset_class.DATASET_ADJENCIES == i).sum(axis=2).sum(axis=1)/2).astype(np.int32) for i in range(1, dataset_class.num_edge_types()+1)]
	
	##################################
	## Number of nodes distribution ##
	##################################
	if show_plots:
		ax = visualize_histogram(data=length_distribution, 
								 bins=log_length_prior.shape[0],
								 xlabel="Number of nodes", 
								 ylabel="Number of graphs",
								 title_text="Node count distribution")
		ax.set_xlim(5, 38)
		plt.tight_layout()
		plt.show()
	length_count = np.bincount(length_distribution)
	print("="*40)
	print("Number of molecules per graph size")
	print("-"*40)
	for i in range(log_length_prior.shape[0]):
		print("Graph size %i: %i" % (i, length_count[i]))
	print("="*40)

	############################
	## Node type distribution ##
	############################
	if show_plots:
		ax = visualize_histogram(data=node_distribution, 
								 bins=np.max(node_distribution)+1,
								 xlabel="Node type", 
								 ylabel="Number of nodes",
								 title_text="Node type distribution",
								 add_stats=False)
		plt.tight_layout()
		plt.show()

	node_count = np.bincount(node_distribution)
	node_log_prob = np.log(node_count) - np.log(node_count.sum())
	print("="*40)
	print("Distribution of node types")
	print("-"*40)
	for i in range(node_log_prob.shape[0]):
		print("Node %i: %6.3f%% (%i) -> %4.2fbpd" % (i, np.exp(node_log_prob[i])*100.0, node_count[i], -(np.log2(np.exp(1))*node_log_prob[i])))
	print("="*40)

	############################
	## Node type distribution ##
	############################
	if show_plots:
		ax = visualize_histogram(data=edge_distribution, 
								 bins=max([np.max(d) for d in edge_distribution])+1,
								 xlabel="Number of edges per type", 
								 ylabel="Number of graphs",
								 title_text="Edge type distribution")
		plt.tight_layout()
		plt.show()

	edge_overall_count = (length_distribution * (length_distribution-1) / 2).sum()
	edge_count = np.array([d.sum() for d in edge_distribution])
	edge_count_sum = edge_count.sum()
	edge_count = np.concatenate([np.array([edge_overall_count-edge_count_sum]), edge_count], axis=0)
	edge_log_prob = np.log(edge_count) - np.log(edge_overall_count)
	print("="*40)
	print("Distribution of edge types")
	print("-"*40)
	for i in range(edge_count.shape[0]):
		print("Edge %i: %4.2f%% (%i) -> %4.2fbpd" % (i, np.exp(edge_log_prob[i])*100.0, edge_count[i], -(np.log2(np.exp(1))*edge_log_prob[i])))
	print("="*40)


def visualize_histogram(data, bins, xlabel, ylabel, title_text, val_range=None, add_stats=True):
	title_font = {'fontsize': 20, 'fontweight': 'bold'}
	axis_font = {'fontsize': 16, 'fontweight': 'medium'}
	ticks_font = {'fontsize': 12, 'fontweight': 'medium'}

	fig, ax = plt.subplots(1, 1, figsize=(10,6))
	if val_range is None:
		val_range = (0, bins-1)
	if isinstance(data, list):
		ax.hist(data, bins=bins, range=val_range, alpha=0.8, label=["data_%i"%i for i in range(len(data))])
	else:
		ax.hist(data, bins=bins, range=val_range, alpha=0.6)
		if add_stats:
			ax.axvline(data.mean(), color='r', linewidth=2, label="Mean", ymax=0.9)
			ax.axvline(np.median(data), color='b', linewidth=2, label="Median", ymax=0.9)
	ax.set_xlabel(xlabel, fontdict=axis_font)
	ax.set_ylabel(ylabel, fontdict=axis_font)
	ax.tick_params(axis='both', labelsize=ticks_font["fontsize"])
	ax.set_title(title_text, fontdict=title_font)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	if add_stats or isinstance(data, list):
		plt.legend()

	return ax


def analyse_generations(dataset_class, nodes, adjacency, length):
	eval_dict, valid_list = dataset_class.evaluate_generations(nodes, adjacency, length=length, full_valid_list=True)
	print("="*50)
	print("Eval dict")
	print("-"*50)
	for key in eval_dict:
		print("%s: %s" % (str(key), str(eval_dict[key])))
	print("Valid list: %s" % str(valid_list[:10]))
	print("-"*50 + "\n")
	node_list = np.concatenate([nodes[i,:length[i]] for i in range(length.shape[0])], axis=0)
	node_distribution = np.bincount(node_list)
	print("="*50)
	print("Predicted node distribution")
	print("-"*50)
	for i in range(node_distribution.shape[0]):
		print("Node %i: %4.2f%% (%i)" % (i, node_distribution[i]*100.0/node_list.shape[0], node_distribution[i]))
	print("-"*50 + "\n")

	edge_list = np.concatenate([adjacency[i,:length[i],:length[i]].reshape(-1) for i in range(length.shape[0])], axis=0)
	edge_distribution = np.bincount(edge_list)
	print("="*50)
	print("Predicted edge distribution")
	print("-"*50)
	for i in range(edge_distribution.shape[0]):
		print("Edge %i: %4.2f%% (%i)" % (i, edge_distribution[i]*100.0/edge_list.shape[0], edge_distribution[i]))
	print("-"*50 + "\n")	
	return valid_list


def find_largest_submolecule(nodes, adjacency):
	bin_adjacency = ((adjacency + np.eye(adjacency.shape[0])) > 0).astype(np.float32)
	length = (nodes >= 0).sum()
	nodes = nodes[:length]
	bin_adjacency = bin_adjacency[:length,:length]

	def _find_coverage(start_node):
		cov_nodes = np.zeros(nodes.shape, dtype=np.float32)
		old_cov = cov_nodes.copy()
		cov_nodes[start_node] = 1
		while np.abs(old_cov-cov_nodes).sum() > 0.0:
			old_cov = cov_nodes.copy()
			cov_nodes = ((bin_adjacency * cov_nodes[None,:]).sum(axis=-1) > 0).astype(np.float32)
		return cov_nodes

	node_coverage = _find_coverage(start_node=0)
	largest_submolecule = np.where(node_coverage)[0]
	while (largest_submolecule.shape[0] < node_coverage.shape[0]-node_coverage.sum()):
		node_idx = [i for i in range(nodes.shape[0]) if node_coverage[i] == 0.0][0]
		sub_coverage = _find_coverage(start_node=node_idx)
		node_coverage = sub_coverage + node_coverage
		sub_molecule = np.where(sub_coverage)[0]
		if sub_molecule.shape[0] > largest_submolecule.shape[0]:
			largest_submolecule = sub_molecule

	nodes = nodes[largest_submolecule]
	adjacency = adjacency[largest_submolecule][:,largest_submolecule]
	return nodes, adjacency

def calculate_node_distribution(dataset_class):
	node_distribution = dataset_class.DATASET_NODES[np.where(dataset_class.DATASET_NODES >= 0)]
	node_count = np.bincount(node_distribution)
	node_log_prob = np.log(node_count) - np.log(node_count.sum())
	return node_log_prob

def calculate_edge_distribution(dataset_class):
	length_distribution = (dataset_class.DATASET_NODES >= 0).sum(axis=1).astype(np.int32)
	edge_distribution = [((dataset_class.DATASET_ADJENCIES == i).sum(axis=2).sum(axis=1)/2).astype(np.int32) for i in range(1, dataset_class.num_edge_types()+1)]
	edge_count = np.array([d.sum() for d in edge_distribution])
	edge_log_prob = np.log(edge_count) - np.log(edge_count.sum())
	return edge_log_prob


def evaluate_generations(dataset_class, nodes, adjacency, length=None, full_valid_list=False, **kwargs):
	global RDKIT_IMPORTED

	if isinstance(nodes, torch.Tensor):
		nodes = nodes.detach().cpu().numpy()
	if isinstance(adjacency, torch.Tensor):
		adjacency = adjacency.detach().cpu().numpy()
	if length is not None and isinstance(length, torch.Tensor):
		length = length.detach().cpu().numpy()

	if not RDKIT_IMPORTED:
		print("Skipped evaluation of generated molecules due to import error...")
		return {}

	eval_dict = {}
	for allow_submolecule in [False, True]:
		if length is not None:
			all_mols = [dataset_class.graph_to_mol(nodes[i,:length[i]], adjacency[i,:length[i],:length[i]], allow_submolecule) for i in range(nodes.shape[0])]
		else:
			all_mols = [dataset_class.graph_to_mol(nodes[i], adjacency[i], allow_submolecule) for i in range(nodes.shape[0])]
		valid = [dataset_class._check_validity(mol) for mol in all_mols]
		binary_valid = [1 if mol is not None else 0 for mol in valid]
		valid = [mol for mol, v in zip(all_mols, binary_valid) if v==1]
		valid_ratio = len(valid)*1.0/len(all_mols)

		valid_smiles = [Chem.MolToSmiles(mol) for mol in valid]
		unique_smiles = list(set(valid_smiles))
		unique_ratio = len(unique_smiles)*1.0/(max(len(valid_smiles), 1e-5))

		novel = [(1 if dataset_class._check_novelty(mol=None, smiles=sm) else 0) for sm in valid_smiles]
		novelty_ratio = sum(novel)*1.0/(max(len(novel), 1e-5))

		inner_eval_dict = {
			"valid_ratio": valid_ratio,
			"unique_ratio": unique_ratio,
			"novelty_ratio": novelty_ratio
		}
		if allow_submolecule:
			inner_eval_dict = {"submol_" + key: inner_eval_dict[key] for key in inner_eval_dict}
		eval_dict.update(inner_eval_dict)

	if full_valid_list:
		return eval_dict, binary_valid
	else:
		return eval_dict


def visualize_molecule(mol, filename="test_img"):
	global RDKIT_IMPORTED, SVG2PDF_IMPORTED
	if not RDKIT_IMPORTED:
		print("[!] WARNING: Skipped visualization of molecules as rdkit is not imported.")
		return
	tmp=AllChem.Compute2DCoords(mol)
	Draw.MolToFile(mol, filename+".svg", size=(400,400))
	if SVG2PDF_IMPORTED:
		cairosvg.svg2pdf(url=filename+".svg", write_to=filename+".pdf")


def calculate_length_prior(nodes):
	length_distribution = (nodes >= 0).sum(axis=1).astype(np.int32)
	length_count = np.bincount(length_distribution)
	length_count = length_count.astype(np.float32) + 1e-5 # Smoothing to prevent log of zero
	log_length_prior = np.log(length_count) - np.log(length_count.sum())
	return log_length_prior