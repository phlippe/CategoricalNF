import torch
import numpy as np 
import random
import torch.utils.data as data
from statistics import mean, median
import math
import json
import urllib.request
import shutil
import zipfile
import os
import sys
sys.path.append("../../../")

from general.mutils import one_hot



class Text8Dataset(data.Dataset):

	VOCABULARY_FILE = "text8/vocabulary.json"
	TOKENFREQ_FILE = "text8/token_freq.npy"
	LENGTH_PROB_FILE = "text8/length_prior.npy"
	VOCAB_DICT = None

	def __init__(self, max_seq_len, train=False, val=False, test=False, root="data/", **kwargs):
		self.max_seq_len = max_seq_len
		self.dist_between_sents = int(self.max_seq_len / 16) if not (val or test) else self.max_seq_len
		self.is_train = train
		self.val = val 
		self.test = test

		filepath = root + "text8/text8.%s.npz" % ("train" if train else "valid" if val else "test")
		if not os.path.isfile(filepath):
			print("[!] WARNING: Could not find %s. Start downloading text8..." % (filepath))
			prepare_text8(root=root)
		self.data = np.load(filepath)["arr_0"].astype(np.long)

		self.vocabulary = Text8Dataset.get_vocabulary(root=root)
		self.index_to_word = {val: key for key, val in self.vocabulary.items()}

		print("Number of tokens", len(self.data))
		print("Dataset length", len(self))


	def __len__(self):
		num_elements = int(math.floor((self.data.shape[0] - self.max_seq_len - self.dist_between_sents) / self.dist_between_sents))
		if self.is_train:
			return num_elements
		elif self.val:
			return min(32768, num_elements)
		else:
			return num_elements


	def __getitem__(self, idx):
		idx = idx * self.dist_between_sents
		if self.is_train:
			idx += np.random.randint(self.dist_between_sents)
		return self.data[idx:idx+self.max_seq_len], self.max_seq_len


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
	def get_vocabulary(root=None, **kwargs):
		if Text8Dataset.VOCAB_DICT is None:
			if root is None:
				vocab_file = Text8Dataset.VOCABULARY_FILE
			else:
				vocab_file = os.path.join(root, Text8Dataset.VOCABULARY_FILE)
			with open(vocab_file, "r") as f:
				Text8Dataset.VOCAB_DICT = json.load(f)
		return Text8Dataset.VOCAB_DICT


	@staticmethod
	def get_torchtext_vocab():
		return None


	@staticmethod
	def get_log_frequency(root="data/"):
		filepath = os.path.join(root, Text8Dataset.TOKENFREQ_FILE)
		if not os.path.isfile(filepath):
			dataset = Text8Dataset(max_seq_len=256, train=True, val=False, test=False, root=root)
			word_bincount = np.bincount(dataset.data)
			print(word_bincount)
			word_probs = word_bincount * 1.0 / word_bincount.sum(keepdims=True)
			word_bincount += 1 # Smoothing
			word_bincount = np.log2(word_bincount.astype(np.float32)) - np.log2(word_bincount.sum(keepdims=True).astype(np.float32))
			for key, val in dataset.vocabulary.items():
				print("%s: %10.8f%%" % (key, (2**word_bincount[val])*100.0))
			print("-"*20)
			print("Uniform Bpc: %4.2f" % (-np.log2(1.0/word_bincount.shape[0])))
			print("Unigram Bpc: %5.4f" % (word_probs * (-word_bincount)).sum())
			print("Vocabulary size: %i" % word_bincount.shape[0])
			np.save(filepath, word_bincount)

		word_bincount = np.load(filepath)
		return word_bincount


def prepare_text8(root="../data/"):
	url = 'http://mattmahoney.net/dc/text8.zip'
	base_filepath = root + "text8/"
	os.makedirs(base_filepath, exist_ok=True)

	filename = base_filepath + 'text8.zip'
	train_size = 90000000
	val_size = 5000000
	test_size = 5000000

	if not os.path.isfile(filename):
		print('Downloading text8 dataset...')

		with urllib.request.urlopen(url) as response, \
			open(filename, 'wb') as outfile:
			shutil.copyfileobj(response, outfile)

	rawdata = zipfile.ZipFile(filename).read('text8').decode('utf-8')

	train_split = rawdata[:train_size]
	valid_split = rawdata[train_size:train_size+val_size]
	test_split = rawdata[train_size+val_size:]

	print("Creating vocabulary...")
	unique_tokens = sorted(list(set(rawdata)))

	vocabulary = {t: i for i, t in enumerate(unique_tokens)}
	vocab_file = root + Text8Dataset.VOCABULARY_FILE
	os.makedirs(vocab_file.rsplit("/",1)[0]+"/", exist_ok=True)
	with open(vocab_file, "w") as f:
		json.dump(vocabulary, f, indent=4)
	print("Exported vocabulary to", vocab_file)


	for data_split, data_name in [(train_split, "train"), (valid_split, "valid"), (test_split, "test")]:
		print("Exporting %s..." % data_name)

		with open(base_filepath + "text8.%s.txt" % data_name, "w") as f:
			f.write(data_split)

		num_data = np.array([vocabulary[t] for t in data_split], dtype=np.uint8)
		np.savez_compressed(base_filepath + "text8.%s.npz" % data_name, num_data)


	print("Creating pretraining directory...")
	pretrain_dir = root + Text8Dataset.PRETRAIN_DIR
	os.makedirs(pretrain_dir, exist_ok=True)
	with open(os.path.join(pretrain_dir, "model_configs.json"), "w") as f:
		f.write("{\n}")
	
	print("Obtaining log frequencies...")
	Text8Dataset.get_log_frequency(root=root)





if __name__ == '__main__':
	np.random.seed(42)
	torch.manual_seed(42)
	
	train_dataset = Text8Dataset(max_seq_len=256, train=True, root="../data/")
	val_dataset = Text8Dataset(max_seq_len=256, val=True, root="../data/")
	test_dataset = Text8Dataset(max_seq_len=256, test=True, root="../data/")
	data_loader = iter(data.DataLoader(train_dataset, 4))

	for data_index in range(10):
		sents = data_loader.next()
		print(sents)
		if data_index > 4:
			break



