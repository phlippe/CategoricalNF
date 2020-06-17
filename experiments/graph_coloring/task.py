import torch
import torch.nn as nn
import numpy as np 
import time
import shutil
import os
import sys
sys.path.append("../../")

from general.task import TaskTemplate
from general.mutils import get_param_val, append_in_dict, get_device, create_channel_mask
from general.parameter_scheduler import *
from layers.flows.distributions import create_prior_distribution

from experiments.graph_coloring.datasets.graph_coloring import GraphColoringDataset
from experiments.graph_coloring.graph_node_flow import GraphNodeFlow
from experiments.graph_coloring.graph_node_rnn import GraphNodeRNN
from experiments.graph_coloring.graph_node_vae import GraphNodeVAE


class TaskGraphColoring(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, batch_size=64):
		super().__init__(model, model_params, load_data=load_data, debug=debug, batch_size=batch_size, name="TaskGraphColoring")

		prior_dist_params = get_param_val(self.model_params, "prior_distribution", dict())
		self.prior_distribution = create_prior_distribution(prior_dist_params)

		self.beta_scheduler = create_scheduler(self.model_params["beta"], "beta")
		self.gamma_scheduler = create_scheduler(self.model_params["gamma"], "gamma")

		self.summary_dict = {"log_prob": list(), "ldj": list(), "z": list(), 
							 "beta": 0, "gamma": 0}
		self.checkpoint_path = None


	def _load_datasets(self):
		self.dataset_class = self.model.dataset_class

		dataset_kwargs = {}
		if isinstance(self.model, GraphNodeRNN):
			graph_ordering = get_param_val(self.model_params, "rnn_graph_ordering", default_val="rand")
			dataset_kwargs["order_graphs"] = graph_ordering

		self.train_dataset = self.dataset_class(train=True, val=False, test=False, **dataset_kwargs)
		self.val_dataset = self.dataset_class(train=False, val=True, test=False, **dataset_kwargs)
		self.test_dataset = self.dataset_class(train=False, val=False, test=True, **dataset_kwargs)


	@staticmethod
	def get_dataset_class(dataset_name):
		size = dataset_name.split("_")[0]
		ncolors = int(dataset_name.split("_")[1])
		GraphColoringDataset.set_dataset(prefix="_"+size, num_colors=ncolors)
		return GraphColoringDataset


	def _train_batch(self, batch, **kwargs):
		batch = self._preprocess_batch(batch)
		if isinstance(self.model, GraphNodeRNN):
			return self._train_batch_rnn(*batch, **kwargs)
		elif isinstance(self.model, GraphNodeVAE):
			return self._train_batch_vae(*batch, **kwargs)
		elif isinstance(self.model, GraphNodeFlow):
			return self._train_batch_flow(*batch, **kwargs)
		

	def _train_batch_rnn(self, x_in, x_adjacency, x_length, x_channel_mask, **kwargs):
		logprob = self.model(x_in, x_adjacency, reverse=False, length=x_length)
		self.summary_dict["log_prob"].append(-logprob.mean().item())
		loss = -logprob.mean()
		return loss


	def _train_batch_vae(self, x_in, x_adjacency, x_length, x_channel_mask, **kwargs):
		logprob, ldj_per_layer = self.model(x_in, x_adjacency, reverse=False, length=x_length)
		self._ldj_per_layer_to_summary(ldj_per_layer)
		self.summary_dict["log_prob"].append(-logprob.mean().item())
		loss = -logprob.mean()
		return loss


	def _train_batch_flow(self, x_in, x_adjacency, x_length, x_channel_mask, iteration=0, **kwargs):
		z, ldj, ldj_per_layer = self.model(x_in, x_adjacency, reverse=False, get_ldj_per_layer=True, 
										   beta=self.beta_scheduler.get(iteration),
										   length=x_length)
		neglog_prob = -(self.prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
		neg_ldj = -ldj

		neg_ldj = neg_ldj / (x_length).float()
		neglog_prob = neglog_prob / (x_length).float()
		loss = (neg_ldj + neglog_prob).mean()

		self.summary_dict["log_prob"].append(neglog_prob.mean().item())
		self.summary_dict["ldj"].append(neg_ldj.mean().item())
		self.summary_dict["beta"] = self.beta_scheduler.get(iteration)

		self._ldj_per_layer_to_summary(ldj_per_layer)

		return loss


	def _eval_batch(self, batch, **kwargs):
		batch = self._preprocess_batch(batch)
		if isinstance(self.model, GraphNodeRNN):
			return self._eval_batch_rnn(*batch, **kwargs)
		elif isinstance(self.model, GraphNodeVAE):
			return self._eval_batch_vae(*batch, **kwargs)
		elif isinstance(self.model, GraphNodeFlow):
			return self._eval_batch_flow(*batch, **kwargs)


	def _eval_batch_rnn(self, x_in, x_adjacency, x_length, x_channel_mask, **kwargs):
		return -self.model(x_in, x_adjacency, reverse=False, length=x_length).mean()


	def _eval_batch_vae(self, x_in, x_adjacency, x_length, x_channel_mask, **kwargs):
		return -self.model(x_in, x_adjacency, reverse=False, length=x_length)[0].mean()


	def _eval_batch_flow(self, x_in, x_adjacency, x_length, x_channel_mask, **kwargs):
		z, ldj, ldj_per_layer = self.model(x_in, x_adjacency, reverse=False, get_ldj_per_layer=True, 
										   length=x_length)
		neglog_prob = -(self.prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
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


	def initialize(self, num_batches=16):
		if not isinstance(self.model, GraphNodeFlow):
			return
		else:
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


	def _verify_permutation(self):
		with torch.no_grad():
			self.model.eval()
			batch = self._get_next_batch()
			batch = TaskTemplate.batch_to_device(batch)
			x_in, x_adjacency, x_length, _ = self._preprocess_batch(batch)
			if hasattr(self.model, "test_permutation"):
				assert self.model.test_permutation(x_in, x_adjacency, x_length), "[!] ERROR: Permutation test failed."
			if hasattr(self.model, "test_reversibility"):
				assert self.model.test_reversibility(x_in, x_adjacency, x_length), "[!] ERROR: Reversibility test failed."


	def _node_sampling(self, temp=1.0, data_loader=None):
		if data_loader is None:
			data_loader = self.val_data_loader
		all_samples = []
		print("Evaluating graph samples...")
		for batch_ind, batch in enumerate(data_loader):
			# Put batch on correct device
			batch = TaskTemplate.batch_to_device(batch)
			_, x_adjacency, x_length, _ = self._preprocess_batch(batch, length_clipping=False)

			z = self.prior_distribution.sample(shape=(x_adjacency.shape[0], x_adjacency.shape[1], self.model.embed_dim),
											   temp=temp).to(x_adjacency.device)
			z, _ = self.model(z, adjacency=x_adjacency, length=x_length, reverse=True)

			all_samples.append((z, x_adjacency, x_length))

			if self.debug and batch_ind > 10:
				break
		return all_samples


	def _eval_finalize_metrics(self, detailed_metrics, is_test=False, initial_eval=False, **kwargs):
		if initial_eval:
			return

		self._verify_permutation()

		start_time = time.time()
		all_samples = self._node_sampling(data_loader=self.val_data_loader if not is_test else self.test_data_loader, 
										  temp=1.0)						 
		end_time = time.time()
		duration = (end_time - start_time)
			
		out = [torch.cat([s[i] for s in all_samples], dim=0).detach().cpu().numpy() for i in range(len(all_samples[0]))]
		z, adjacency, length = out[0], out[1], out[2]

		num_graphs = z.shape[0]
		print("Generated %i graphs in %5.2fs => %4.3fs per graph" % (num_graphs, duration, duration/num_graphs))
		gen_eval_dict = self.dataset_class.evaluate_generations(nodes=z, adjacency=adjacency, length=length)
		detailed_metrics.update(gen_eval_dict)
		print("Validity ratio: %4.2f%%" % (100.0*gen_eval_dict["valid_ratio"]))
		
		if self.checkpoint_path is not None:
			np.savez_compressed(os.path.join(self.checkpoint_path, "node_samples.npz"), z=z, adjacency=adjacency, length=length)


	def add_summary(self, writer, iteration, checkpoint_path=None):
		self.checkpoint_path = checkpoint_path
		super().add_summary(writer, iteration, checkpoint_path=checkpoint_path)


	def export_best_results(self, checkpoint_path, iteration):
		if os.path.isfile(os.path.join(checkpoint_path, "node_samples.npz")):
			shutil.copy(os.path.join(checkpoint_path, "node_samples.npz"), 
						os.path.join(checkpoint_path, "node_samples_best.npz"))