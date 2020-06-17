import torch
import torch.nn as nn
import numpy as np 
import time
import shutil
import os
import sys
sys.path.append("../../")

from general.task import TaskTemplate
from general.mutils import get_param_val, append_in_dict, get_device, create_channel_mask, debug_level
from general.parameter_scheduler import *
from layers.flows.distributions import create_prior_distribution

from experiments.molecule_generation.datasets.zinc250k import Zinc250kDataset
from experiments.molecule_generation.datasets.moses import MosesDataset
from experiments.molecule_generation.graphCNF import GraphCNF


class TaskMoleculeGeneration(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, batch_size=64):
		super().__init__(model, model_params, load_data=load_data, debug=debug, batch_size=batch_size, name="TaskMoleculeGeneration")

		prior_dist_params = get_param_val(self.model_params, "prior_distribution", dict())
		self.prior_distribution = create_prior_distribution(prior_dist_params)

		self.beta_scheduler = create_scheduler(self.model_params["beta"], "beta")

		self.summary_dict = {"log_prob": list(), "ldj": list(),
							 "beta": 0}
		self.checkpoint_path = None


	def _load_datasets(self):
		self.dataset_class = self.model.dataset_class
		self.train_dataset = self.dataset_class(train=True, val=False, test=False, data_root="data/")
		self.val_dataset = self.dataset_class(train=False, val=True, test=False, data_root="data/")
		self.test_dataset = self.dataset_class(train=False, val=False, test=True, data_root="data/")
		self.log_length_prior = self.dataset_class.get_length_prior()


	@staticmethod
	def get_dataset_class(dataset_name):
		if dataset_name.lower() == "zinc250k":
			dataset_class = Zinc250kDataset
		elif dataset_name == "moses":
			dataset_class = MosesDataset
		else:
			assert False, "[!] ERROR: Unknown dataset \"%s\"" % (dataset_name)
		return dataset_class


	def _train_batch(self, batch, iteration=0):
		x_in, x_adjacency, x_length, x_channel_mask = self._preprocess_batch(batch)
		z_nodes, ldj, ldj_per_layer = self.model(x_in, x_adjacency, reverse=False, get_ldj_per_layer=True, 
												 beta=self.beta_scheduler.get(iteration),
												 length=x_length)
		neglog_prob = -(self.prior_distribution.log_prob(z_nodes) * x_channel_mask).sum(dim=[1,2])
		neg_ldj = -ldj

		neg_ldj = neg_ldj / (x_length).float()
		neglog_prob = neglog_prob / (x_length).float()
		loss = (neg_ldj + neglog_prob).mean()

		self.summary_dict["log_prob"].append(neglog_prob.mean().item())
		self.summary_dict["ldj"].append(neg_ldj.mean().item())
		self.summary_dict["beta"] = self.beta_scheduler.get(iteration)
		self._ldj_per_layer_to_summary(ldj_per_layer)

		return loss


	def _eval_batch(self, batch, is_test=False):
		x_in, x_adjacency, x_length, x_channel_mask = self._preprocess_batch(batch)
		z_nodes, ldj, ldj_per_layer = self.model(x_in, x_adjacency, reverse=False, get_ldj_per_layer=True, 
												 beta=1.0,
												 length=x_length)
		neglog_prob = -(self.prior_distribution.log_prob(z_nodes) * x_channel_mask).sum(dim=[1,2])
		neg_ldj = -ldj
		
		neg_ldj = neg_ldj / (x_length).float()
		neglog_prob = neglog_prob / (x_length).float()
		loss = (neg_ldj + neglog_prob).mean()

		return loss


	def _preprocess_batch(self, batch, length_clipping=True):
		x_in, x_adjacency, x_length = batch
		if length_clipping:
			max_len = x_length.max()
			x_in = x_in[:,:max_len].contiguous()
			x_adjacency = x_adjacency[:,:max_len,:max_len].contiguous()
		x_channel_mask = create_channel_mask(x_length, max_len=x_in.shape[1])
		return x_in, x_adjacency, x_length, x_channel_mask


	def add_summary(self, writer, iteration, checkpoint_path=None):
		super().add_summary(writer, iteration, checkpoint_path)
		self.checkpoint_path = checkpoint_path


	def initialize(self, num_batches=16):
		if self.model.need_data_init():
			print("Preparing data dependent initialization...")
			batch_list = []
			for _ in range(num_batches):
				batch = self._get_next_batch()
				batch = TaskTemplate.batch_to_device(batch)
				x_in, x_adjacency, x_length, _ = self._preprocess_batch(batch, length_clipping=False)
				batch_tuple = (x_in, {"length": x_length, "adjacency": x_adjacency})
				batch_list.append(batch_tuple)
			self.model.initialize_data_dependent(batch_list)
		with torch.no_grad():
			self._verify_permutation()
			self.sample(sample_size=2)


	def _verify_permutation(self):
		with torch.no_grad():
			self.model.eval()
			batch = self._get_next_batch()
			batch = TaskTemplate.batch_to_device(batch)
			x_in, x_adjacency, x_length, _ = self._preprocess_batch(batch)
			self.model.test_reversibility(x_in, x_adjacency, x_length)


	def sample(self, sample_size=16, temp=1.0):
		with torch.no_grad():
			z_nodes = self.prior_distribution.sample(shape=(sample_size, self.model.max_num_nodes, self.model.encoding_dim_nodes),
													 temp=temp).to(get_device())
			
			length_prior = torch.from_numpy(self.log_length_prior).to(get_device()).exp()
			length = torch.multinomial(input=length_prior, num_samples=1, replacement=True)[0]
			min_len = (length-2).clamp(min=0)
			max_len = (length+3).clamp(max=length_prior.shape[0])
			pruned_length_prior = length_prior[min_len:max_len]
			pruned_length_prior = pruned_length_prior / pruned_length_prior.sum().clamp(min=1e-7)
			length = min_len + torch.multinomial(input=pruned_length_prior, num_samples=sample_size, replacement=True)
			
			z_nodes = z_nodes[:,:length.max()].contiguous()

			z_out, _ = self.model(z_nodes, reverse=True, get_ldj_per_layer=False, length=length, sample_temp=temp)
			z, adjacency = z_out
			# Padding
			if z.shape[1] < self.model.max_num_nodes:
				z = torch.cat([z, z.new_zeros((z.shape[0], self.model.max_num_nodes-z.shape[1])+z.shape[2:])], dim=1)
			if adjacency.shape[1] < self.model.max_num_nodes:
				adjacency = torch.cat([adjacency, 
						adjacency.new_zeros((adjacency.shape[0], self.model.max_num_nodes-adjacency.shape[1])+adjacency.shape[2:])], 
					dim=1)
				adjacency = torch.cat([adjacency, 
						adjacency.new_zeros(adjacency.shape[:2] + (self.model.max_num_nodes-adjacency.shape[2],))], 
					dim=2)
		return (z, adjacency, length)


	def graph_sampling(self, temp=1.0):
		all_samples = []
		if torch.cuda.is_available():
			num_gpus = torch.cuda.device_count() if isinstance(self.model, nn.DataParallel) else 1
			num_batches, batch_size = 8 if self.debug else int(80/num_gpus), int(128 * num_gpus)
		else:
			num_batches, batch_size = 1, 4
		print("Sampling graphs (%i batches, %i batch size)..." % (num_batches, batch_size))
		
		for batch_ind in range(num_batches):
			if debug_level() == 0:
				print("Sampling process: %4.2f%%" % (100.0 * batch_ind / num_batches), end="\r")
			s = self.sample(sample_size=batch_size, temp=temp)
			all_samples.append(s)

		return all_samples


	def _eval_finalize_metrics(self, detailed_metrics, is_test=False, perform_full_eval=False, initial_eval=False):
		if initial_eval:
			return

		self._verify_permutation()

		start_time = time.time()
		all_samples = self.graph_sampling(temp=1.0)						 
		end_time = time.time()
		duration = (end_time - start_time)
			
		out = [torch.cat([s[i] for s in all_samples], dim=0).detach().cpu().numpy() for i in range(len(all_samples[0]))]
		z, adjacency, length = out[0], out[1], out[2]

		num_graphs = z.shape[0]
		print("Generated %i graphs in %5.2fs => %4.3fs per graph" % (num_graphs, duration, duration/num_graphs))
		gen_eval_dict = self.dataset_class.evaluate_generations(nodes=z, adjacency=adjacency, length=length)
		detailed_metrics.update(gen_eval_dict)
		print("Validity ratio: %4.2f%%" % (100.0*gen_eval_dict["valid_ratio"]))
		detailed_metrics["loss_metric"] = -gen_eval_dict["valid_ratio"] # Negative as we want to maximize it while loss is minimized
		if self.checkpoint_path is not None:
			np.savez_compressed(os.path.join(self.checkpoint_path, "molecule_samples.npz"), z=z, adjacency=adjacency, length=length)


	def export_best_results(self, checkpoint_path, iteration):
		if os.path.isfile(os.path.join(checkpoint_path, "molecule_samples.npz")):
			shutil.copy(os.path.join(checkpoint_path, "molecule_samples.npz"), 
						os.path.join(checkpoint_path, "molecule_samples_best.npz"))