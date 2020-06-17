import torch
import numpy as np 
import random
import torch.utils.data as data
from torchnlp.datasets import penn_treebank_dataset
import math
import json
import os
import sys
sys.path.append("../../../")

from general.mutils import one_hot



class PennTreeBankDataset(data.Dataset):

	VOCABULARY_FILE = "penn-treebank/vocabulary.json"
	TOKENFREQ_FILE = "penn-treebank/token_freq.npy"
	LENGTH_PROB_FILE = "penn-treebank/length_prior.npy"
	VOCAB_DICT = None

	def __init__(self, max_seq_len, train=False, val=False, test=False, root="data/", **kwargs):
		self.max_seq_len = max_seq_len
		self.dist_between_sents = int(self.max_seq_len / 10)
		self.is_train = train

		dataset = penn_treebank_dataset(root + "penn-treebank", train=train, dev=val, test=test)

		self.vocabulary = PennTreeBankDataset.get_vocabulary(root=root)
		self.index_to_word = {val: key for key, val in self.vocabulary.items()}

		words = [[]]
		for word_index, word in enumerate(dataset):
			if word == "</s>":
				words.append([])
			else:
				if word in self.vocabulary:
					words[-1].append(self.vocabulary[word])
				else:
					words[-1] += [self.vocabulary[c] for c in word]
				if word != "</s>":
					words[-1].append(self.vocabulary[" "])
				
		self.data = [np.array(sent) for sent in words if (len(sent) != 0 and len(sent)<self.max_seq_len)]

		print("Length of dataset: ", len(self))


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		length = self.data[idx].shape[0]
		if length < self.max_seq_len:
			padded_data = np.concatenate([self.data[idx], np.zeros(self.max_seq_len-length, dtype=np.int32)], axis=0)
		else:
			padded_data = self.data[idx][:self.max_seq_len]
			length = min(self.max_seq_len, length)
		return padded_data, length


	def tensor_to_text(self, tensor):
		if isinstance(tensor, tuple) or isinstance(tensor, list):
			tensor, length = tensor
		else:
			length = tensor.new_zeros(tensor.size(0),dtype=torch.long) + tensor.size(1)
		tensor = tensor.detach().cpu().numpy()
		num_sents, _ = tensor.shape
		sentences = []
		for sent_index in range(num_sents):
			sent = ""
			for token_index in range(length[sent_index]):
				voc_index = tensor[sent_index, token_index]
				if voc_index not in self.index_to_word:
					print("[%] WARNING: During conversion from tensor to text, index %i was found which is not in the vocabulary. Replaced with \"?\"")
					sent += "?"
				else:
					sent += self.index_to_word[voc_index]
			sentences.append(sent)
		return sentences


	def get_embedding_vectors(self):
		return None


	@staticmethod
	def get_vocabulary(root="data/", **kwargs):
		if PennTreeBankDataset.VOCAB_DICT is None:
			if root is None:
				vocab_file = PennTreeBankDataset.VOCABULARY_FILE
			else:
				vocab_file = os.path.join(root, PennTreeBankDataset.VOCABULARY_FILE)
			if not os.path.isfile(vocab_file):
				PennTreeBankDataset.create_vocabulary(root=root)
			with open(vocab_file, "r") as f:
				PennTreeBankDataset.VOCAB_DICT = json.load(f)
		return PennTreeBankDataset.VOCAB_DICT


	@staticmethod
	def get_torchtext_vocab():
		return None


	@staticmethod
	def create_vocabulary(root="data/"):
		if root is None:
			root = ""
		dataset = penn_treebank_dataset(root + "penn-treebank", train=True, dev=False, test=False)
		all_words = [w for w in dataset]
		vocabulary = list(set([c for w in all_words for c in w])) + [" ", "<unk>", "</s>"]
		vocabulary = sorted(vocabulary)
		vocabulary = {vocabulary[i]: i for i in range(len(vocabulary))}
		with open(root + PennTreeBankDataset.VOCABULARY_FILE, "w") as f:
			json.dump(vocabulary, f, indent=4)


	@staticmethod
	def get_log_frequency(root="data/"):
		filepath = os.path.join(root, PennTreeBankDataset.TOKENFREQ_FILE)
		if True or not os.path.isfile(filepath):
			dataset = PennTreeBankDataset(max_seq_len=1000, train=True, val=False, test=False, root=root)
			word_bincount = np.bincount(np.concatenate(dataset.data, axis=0))
			print(word_bincount)
			word_probs = word_bincount * 1.0 / word_bincount.sum(keepdims=True)
			word_bincount += 1 # Smoothing
			word_bincount = np.log2(word_bincount.astype(np.float32)) - np.log2(word_bincount.sum(keepdims=True).astype(np.float32))
			for key, val in dataset.vocabulary.items():
				print("%s: %10.8f%%" % (key, 2**(word_bincount[val])*100.0))
			print("-"*20)
			print("Uniform Bpc: %4.2f" % (-np.log2(1.0/word_bincount.shape[0])))
			print("Unigram Bpc: %5.4f" % (word_probs * (-word_bincount)).sum())
			np.save(filepath, word_bincount)

		word_bincount = np.load(filepath)
		return word_bincount


	@staticmethod
	def get_length_prior(max_seq_len, root="data/"):
		file_path = os.path.join(root, PennTreeBankDataset.LENGTH_PROB_FILE)
		if not os.path.isfile(file_path):
			train_dataset = PennTreeBankDataset(root=root, max_seq_len=1000, train=True)
			val_dataset = PennTreeBankDataset(root=root, max_seq_len=1000, val=True)
			sent_lengths = [d.shape[0] for d in train_dataset.data] + [d.shape[0] for d in val_dataset.data]
			sent_lengths_freq = np.bincount(np.array(sent_lengths))
			np.save(file_path, sent_lengths_freq)

		length_prior_count = np.load(file_path)
		length_prior_count = length_prior_count[:max_seq_len+1] + 1
		log_length_prior = np.log(length_prior_count) - np.log(length_prior_count.sum())
		return log_length_prior
		

if __name__ == '__main__':
	np.random.seed(42)
	PennTreeBankDataset.get_log_frequency(root="../data/")
	dataset = PennTreeBankDataset(max_seq_len=288, train=True, val=False, test=False, sentence_wise=True, root="../data/")
	data_loader = iter(data.DataLoader(dataset, 4))

	for data_index in range(10):
		sents = data_loader.next()
		print(sents)
		if data_index > 4:
			break


