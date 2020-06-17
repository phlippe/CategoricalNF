import torch
import torch.utils.data as data
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from statistics import mean, median, stdev


class BucketSampler(data.Sampler):

	def __init__(self, dataset, batch_size, len_step=1):
		super().__init__(dataset)
		self.dataset = dataset
		self.batch_size = batch_size
		self.len_step = len_step
		self._prepare()

	def _prepare(self):
		indices = self.dataset.data_indices
		lengths = (self.dataset.__class__.DATASET_NODES[indices] >= 0).sum(axis=-1)
		lengths = lengths // self.len_step
		linear_indices = np.arange(indices.shape[0]).astype(np.int32)
		self.unique_lengths = np.unique(lengths)
		self.indices_by_lengths = [linear_indices[lengths==l] for l in self.unique_lengths]

	def __iter__(self):
		sampled_indices = []
		ind_by_len = [np.random.permutation(inds) for inds in self.indices_by_lengths]

		while len(sampled_indices) < len(self):
			p = [inds.shape[0] for inds in ind_by_len]
			p = [e*1.0/sum(p) for e in p]
			global_len = np.random.choice(len(ind_by_len), p=p, size=1)[0]

			global_inds = []

			def add_len(global_inds, local_len):
				size_to_add = self.batch_size - len(global_inds)
				global_inds += ind_by_len[local_len][:size_to_add].tolist()
				if ind_by_len[local_len].shape[0] > size_to_add:
					ind_by_len[local_len] = ind_by_len[local_len][size_to_add:]
				else:
					ind_by_len[local_len] = np.array([])
				return global_inds

			add_len(global_inds, global_len)

			while len(global_inds) < self.batch_size:
				if all([inds.shape[0]==0 for inds in ind_by_len]):
					break
				global_len = (global_len + 1) % len(ind_by_len)
				add_len(global_inds, global_len)

			sampled_indices += global_inds


		return iter(sampled_indices)


	def __len__(self):
		return len(self.dataset)