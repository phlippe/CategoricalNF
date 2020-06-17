import torch
import numpy as np 
import random
import torch.utils.data as data
import math
from collections import Counter
from copy import copy
import sys
sys.path.append("../../../")

"""
Dataset class for creating the set summation dataset. Given a sum L and value range 1-N, we create all valid sets.
Thereby we sample the sets with a frequency proportional to their number of permutations.
"""


class SetSummationDataset(data.Dataset):


	def __init__(self, set_size, max_sum=42, train=True, val=False, test=False, **kwargs):
		self.set_size = set_size
		self.num_classes = set_size
		self.train = train

		self.set_addition = create_all_examples(set_size=self.set_size, min_num=1, max_num=self.set_size, max_sum=max_sum)
		set_permuts = calc_permuts_per_examples(self.set_addition)
		set_all_exmp = sum(set_permuts)
		self.set_probs = [p*1.0/set_all_exmp for p in set_permuts] # Sampling prob proportional to number of permutations

		if not self.train:
			# For the validation and test set, we need generate a fixed sample set beforehand
			np.random.seed(123 if val else 101)
			idx_samples = np.random.choice(len(self.set_addition), p=self.set_probs, size=(32768,))
			set_exmps = [copy(self.set_addition[idx]) for idx in idx_samples]
			[random.shuffle(exmp) for exmp in set_exmps]
			self.set_addition = set_exmps
		self.set_addition = np.array(self.set_addition).astype(np.int64)


	def __len__(self):
		return self.set_addition.shape[0]


	def __getitem__(self, idx):
		idx_set = self.set_addition[idx]
		if self.train:
			idx_set = idx_set[np.random.permutation(self.set_size)]
		return idx_set-1 # To shift 1-16 to 0-15


	def get_sampler(self, batch_size, drop_last=False, **kwargs):
		return data.BatchSampler(PreSampler(self), batch_size, drop_last=drop_last)


	@staticmethod
	def get_vocab_size(set_size):
		return set_size


# We use a sampler here to have a single "np.random.choice" instead in every __get_item___ call
class PreSampler(data.Sampler):

	def __init__(self, dataset):
		super().__init__(dataset)
		self.dataset = dataset
		if self.dataset.train:
			self.sample_size = 16 * len(self.dataset) # Arbitrary length, number of samples per "epoch"
		else:
			self.sample_size = len(self.dataset)

	def __iter__(self):
		if self.dataset.train:
			indices = np.random.choice(self.dataset.set_addition.shape[0], p=self.dataset.set_probs, size=(self.sample_size,))
		else:
			# The test set already sampled with the frequency
			indices = np.arange(len(self.dataset), dtype=np.int32)
		return iter(indices.tolist())

	def __len__(self):
		return self.sample_size



def create_all_examples(set_size, min_num=1, max_num=9, max_sum=-1):
	# Function for creating all valid sets for a certain sum and value range
	# Efficiency/Speed could be improved by caching lower set sums if needed

	if max_sum == -1:
		max_sum = int(set_size * (max_num + min_num)/2)

	all_examples = []

	def recurs_fun(exmp):
		if len(exmp) == set_size:
			if max_sum < 0 or sum(exmp) == max_sum:
				all_examples.append(exmp)
			return

		current_sum, spots_to_fill = sum(exmp), set_size-len(exmp)
		if len(exmp)>0:
			max_exmp = exmp[-1]
		else:
			max_exmp = min(max_num, max_sum-current_sum-spots_to_fill+1)
		for k in range(max_exmp, min_num-1, -1):
			new_exmp = exmp + [k]
			if max_sum < 0 or sum(new_exmp) <= max_sum:
				recurs_fun(new_exmp)
		return

	recurs_fun([])

	if max_sum > 0:
		for exmp in all_examples:
			assert sum(exmp) == max_sum

	return all_examples

def calc_permuts_per_examples(all_examples):
	num_perm = []
	for exmp in all_examples:
		count = Counter(exmp).values()
		num_perm.append(math.factorial(len(exmp)) / np.prod([math.factorial(int(c)) for c in count]))
	return num_perm

def calc_optimum(set_size, all_examples=None):
	if all_examples is None:
		all_examples = create_all_examples(set_size)
	num_perm = 0
	for exmp in all_examples:
		count = Counter(exmp).values()
		num_perm += math.factorial(len(exmp)) / np.prod([math.factorial(int(c)) for c in count])
	print("Number of permutations of valid sets:", num_perm)
	return (np.log2(num_perm))/set_size

def calc_random(vocab_size, **kwargs):
	return np.log2(vocab_size)


if __name__ == '__main__':
	np.random.seed(42)
	set_size = 16
	min_num, max_num = 1, 16
	max_sum = 42
	vocab_size = max_num - min_num + 1

	print("Finding valid sets for L=%i, set size %i and value range %i-%i..." % (max_sum, set_size, min_num, max_num))

	all_examples = create_all_examples(set_size=set_size, min_num=min_num, max_num=max_num, max_sum=max_sum)
	print("Number of samples:", len(all_examples))
	if len(all_examples) < 100:
		print("Samples:", all_examples)
	else:
		print("Samples:")
		smp_idx = np.random.randint(len(all_examples), size=(10,))
		for idx in smp_idx:
			print(all_examples[idx])
	print("Number of random value assignments overall:", vocab_size**set_size)
	print("Optimum: %3.2fbpd" % (calc_optimum(set_size=set_size, all_examples=all_examples)))
	print("Random baseline: %3.2fbpd" % (calc_random(vocab_size=vocab_size)))