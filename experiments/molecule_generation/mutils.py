import torch
import numpy as np


def get_adjacency_indices(num_nodes, length):
	x_indices1 = torch.LongTensor([i for i in range(num_nodes) for j in range(i+1, num_nodes)]).to(length.device)
	x_indices2 = torch.LongTensor([j for i in range(num_nodes) for j in range(i+1, num_nodes)]).to(length.device)

	mask_valid = (x_indices1[None,:] < length[:,None]).float() * (x_indices2[None,:] < length[:,None]).float()
	return mask_valid, (x_indices1, x_indices2)

def adjacency2pairs(adjacency, length):
	num_nodes = adjacency.shape[1]
	mask_valid, x_indices = get_adjacency_indices(num_nodes, length=length)
	x_indices1, x_indices2 = x_indices

	adjacency = adjacency.view(adjacency.shape[0], num_nodes*num_nodes)
	x_flat_indices = x_indices1 + x_indices2*num_nodes
	edge_pairs = adjacency.index_select(index=x_flat_indices, dim=1)
	
	return edge_pairs, (x_indices1, x_indices2), mask_valid

def pairs2adjacency(num_nodes, pairs, length, x_indices):
	x_indices1, x_indices2 = x_indices
	x_flat_indices = x_indices1 + x_indices2*num_nodes
	x_flat_indices_reversed = x_indices2*num_nodes + x_indices1
	_, sort_indices = x_flat_indices.sort(0, descending=False)
	adjacency = pairs.new_zeros(pairs.size(0), num_nodes*num_nodes)
	for b in range(pairs.shape[0]):
		adjacency[b,x_flat_indices] = pairs[b]
	adjacency = adjacency.view(pairs.size(0), num_nodes, num_nodes)
	adjacency = adjacency + torch.transpose(adjacency, 1, 2)
	return adjacency.long()