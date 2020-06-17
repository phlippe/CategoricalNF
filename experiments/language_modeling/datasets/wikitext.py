import torch
import numpy as np 
import random
import torch.utils.data as data
import torchtext
from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer
import torchtext.data as textdata

from statistics import mean, median
import math
import json
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../../../")

from general.mutils import one_hot



class WikiTextDataset(data.Dataset):

	DATASET_SETS = None
	DATASET_VOCAB = None
	TEXT = None


	def __init__(self, max_seq_len, train=False, val=False, test=False, root="data/", **kwargs):
		self.max_seq_len = max_seq_len
		self.dist_between_sents = int(self.max_seq_len / 10)

		WikiTextDataset._load_datasets(root=root)

		dataset = None
		if train:
			dataset = WikiTextDataset.DATASET_SETS["train"]
		elif val:
			dataset = WikiTextDataset.DATASET_SETS["val"]
		elif test:
			dataset = WikiTextDataset.DATASET_SETS["test"]
		else:
			print("[!] ERROR: No dataset split specified for WikiText dataset.")
			sys.exit(1)
		self.data = WikiTextDataset.TEXT.numericalize([dataset[0].text]).squeeze(dim=0) # Shape: [NUM_WORDS]
		
		self.is_train = train
		self.vocabulary = WikiTextDataset.DATASET_VOCAB.stoi
		self.index_to_word = {val: key for key, val in self.vocabulary.items()}
		self.index_to_word[0] = "<unk>"

		print("Vocabulary size", len(self.index_to_word.keys()))
		print("Dataset size", len(self))


	def __len__(self):
		return int(math.floor((self.data.shape[0] - self.max_seq_len - self.dist_between_sents) / self.dist_between_sents))


	def __getitem__(self, idx):
		idx = idx * self.dist_between_sents
		if self.is_train:
			idx += np.random.randint(self.dist_between_sents)
		return self.data[idx:idx+self.max_seq_len]


	def get_embedding_vocab(self):
		return WikiTextDataset.DATASET_VOCAB


	@staticmethod
	def get_vocabulary(**kwargs):
		WikiTextDataset._load_datasets()
		return WikiTextDataset.DATASET_VOCAB.stoi


	@staticmethod
	def get_torchtext_vocab():
		WikiTextDataset._load_datasets()
		return WikiTextDataset.DATASET_VOCAB


	@staticmethod
	def _load_datasets(root="data/", small_dataset=False):
		if WikiTextDataset.DATASET_SETS is None:
			TEXT = textdata.Field(lower=True, batch_first=True)
			dataset_class = WikiText103 if not small_dataset else WikiText2
			print("Loading WikiText%s datasets..." % ("2" if small_dataset else "103"))
			train, val, test = dataset_class.splits(root=root, text_field=TEXT)
			WikiTextDataset.DATASET_SETS = {"train": train, "val": val, "test": test}
			WikiTextDataset.TEXT = TEXT

		if WikiTextDataset.DATASET_VOCAB is None:
			print("Loading GloVe embeddings...")
			vocab_vectors = torchtext.vocab.GloVe(name="840B", cache=root + "pretrained_embed", dim=300)
			print("Building vocabulary...")
			WikiTextDataset.TEXT.build_vocab(WikiTextDataset.DATASET_SETS["train"], max_size=10000, vectors=vocab_vectors)
			WikiTextDataset.DATASET_VOCAB = WikiTextDataset.TEXT.vocab


	@staticmethod
	def get_log_frequency(root="data/"):
		WikiTextDataset._load_datasets(root=root)
		data = WikiTextDataset.TEXT.numericalize([WikiTextDataset.DATASET_SETS["train"][0].text]).squeeze(dim=0)
		word_bincount = np.bincount(data.cpu().numpy())
		word_bincount += 1 # Smoothing
		print("Maximum word count: %i" % (word_bincount.max()))
		print("Minimum word count: %i" % (word_bincount.min()))
		print(word_bincount[-10:])
		word_probs = word_bincount * 1.0 / word_bincount.sum(keepdims=True)
		word_bincount = np.log2(word_bincount.astype(np.float32)) - np.log2(word_bincount.sum(keepdims=True).astype(np.float32))
		printed_zero = False
		for key in WikiTextDataset.DATASET_VOCAB.stoi.keys():
			val = WikiTextDataset.DATASET_VOCAB.stoi[key]
			if (val == 0 and not printed_zero) or \
			   (val == 1) or \
			   (val != 0 and word_probs[val] > 0.001) or \
			   (val > 10000 and val < 10010) or \
			   (val > 15000 and val < 15010) or \
			   (val > 19900):
				printed_zero = printed_zero or (val == 0)
				print("%s (%s): %4.2f%%, log %4.2f" % (key, str(val), word_probs[val]*100.0, word_bincount[val]))

		print("-"*20)
		print("Unigram Bpc: %4.2f" % (word_probs * (-word_bincount)).sum())
		return word_bincount


