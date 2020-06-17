import torch
import numpy as np 
import random
import torch.utils.data as data
import sys
sys.path.append("../../../")


"""
Dataset class for creating the shuffling dataset. 
"""


class SetShufflingDataset(data.Dataset):


	def __init__(self, set_size, train=True, val=False, test=False, **kwargs):
		self.set_size = set_size
		self.num_classes = set_size

		self.shuffle_set = None

		if val or test:
			np.random.seed(123 if val else 101)
			num_shuffles = 32768
			self.shuffle_set = np.stack([self._generate_shuffle() for _ in range(num_shuffles)])


	def __len__(self):
		return int(1e8) if self.shuffle_set is None else self.shuffle_set.shape[0]


	def __getitem__(self, idx):
		if self.shuffle_set is None:
			return self._generate_shuffle()
		else:
			return self.shuffle_set[idx]


	def _generate_shuffle(self):
		# For permutation-invariant models, shuffling the elements does not make a difference
		# We apply it here for safety
		return np.random.permutation(self.set_size)


	@staticmethod
	def get_vocab_size(set_size):
		return set_size
	
def calc_optimum(seq_len):
	# The optimal distribution can be expressed as an autoregressive:
	#  Given first N numbers, the next one can be one out of seq_len-N with a uniform distribution
	#  => log2(seq_len-N)
	class_bpd = sum([np.log2(i) for i in range(1,seq_len+1)])/seq_len
	return class_bpd

def calc_random(seq_len):
	return np.log2(seq_len)

if __name__ == '__main__':
	for seq_len in [2, 3, 4, 8, 16, 32, 64, 128]:
		print("Optimum for sequence length %i: %5.4f vs %5.4f (random)" % ( seq_len, calc_optimum(seq_len), calc_random(seq_len) ) )