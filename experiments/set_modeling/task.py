import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append("../../")

from general.task import TaskTemplate
from general.mutils import get_param_val, append_in_dict, get_device, create_channel_mask
from general.parameter_scheduler import *
from layers.flows.distributions import create_prior_distribution

from experiments.set_modeling.datasets.set_shuffling import SetShufflingDataset
from experiments.set_modeling.datasets.set_summation import SetSummationDataset
from experiments.set_modeling.discrete_flow import DiscreteFlowSetModeling


class TaskSetModeling(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, batch_size=64):
		super().__init__(model, model_params, load_data=load_data, debug=debug, batch_size=batch_size, name="TaskSetModeling")

		prior_dist_params = get_param_val(self.model_params, "prior_distribution", allow_default=False, error_location="TaskSetModeling - init")
		self.prior_distribution = create_prior_distribution(prior_dist_params)

		self.beta_scheduler = create_scheduler(self.model_params["beta"], "beta")
		
		self.summary_dict = {"log_prob": list(), "ldj": list(), "z": list(),
							 "beta": 0}


	def _load_datasets(self):
		self.set_size = get_param_val(self.model_params, "set_size", allow_default=False)

		dataset_name = get_param_val(self.model_params, "dataset", default_val="shuffling")
		dataset_class, dataset_kwargs = TaskSetModeling.get_dataset_class(dataset_name, return_kwargs=True)
		print("Loading dataset %s..." % dataset_name)

		self.train_dataset = dataset_class(set_size=self.set_size, train=True, **dataset_kwargs)
		self.val_dataset = dataset_class(set_size=self.set_size, val=True, **dataset_kwargs)
		self.test_dataset = dataset_class(set_size=self.set_size, test=True, **dataset_kwargs)
		

	@staticmethod
	def get_dataset_class(dataset_name, return_kwargs=False):
		dataset_kwargs = {}
		if dataset_name == "shuffling":
			dataset_class = SetShufflingDataset
		elif dataset_name.startswith("summation"):
			dataset_class = SetSummationDataset
			if "#" in dataset_name:
				dataset_kwargs["max_sum"] = int(dataset_name.split("#")[1])
		else:
			assert False, "[!] ERROR: Unknown dataset class \"%s\"" % (dataset_name)

		if return_kwargs:
			return dataset_class, dataset_kwargs
		else:
			return dataset_class


	def _train_batch(self, batch, iteration=0):
		x_in, x_length, x_channel_mask = self._preprocess_batch(batch)
		if isinstance(self.model, DiscreteFlowSetModeling):
			return self._train_batch_discrete(x_in, x_length)
		else:
			z, ldj, ldj_per_layer = self.model(x_in, reverse=False, get_ldj_per_layer=True, 
											   beta=self.beta_scheduler.get(iteration),
											   length=x_length)
			neglog_prob = -(self.prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
			neg_ldj = -ldj
			
			loss, neg_ldj, neglog_prob = self._calc_loss(neg_ldj, neglog_prob, x_length)

			self.summary_dict["log_prob"].append(neglog_prob.item())
			self.summary_dict["ldj"].append(neg_ldj.item())
			self.summary_dict["beta"] = self.beta_scheduler.get(iteration)
			self._ldj_per_layer_to_summary(ldj_per_layer)

			return loss


	def _train_batch_discrete(self, x_in, x_length):
		_, ldj = self.model(x_in, reverse=False, length=x_length)
		loss = (-ldj / x_length.float()).mean()#
		return loss


	def _eval_batch(self, batch, is_test=False):
		x_in, x_length, x_channel_mask = self._preprocess_batch(batch)
		if isinstance(self.model, DiscreteFlowSetModeling):
			return self._eval_batch_discrete(x_in, x_length)
		else:
			z, ldj = self.model(x_in, reverse=False, get_ldj_per_layer=False, beta=1, 
								length=x_length)
			neglog_prob = -(self.prior_distribution.log_prob(z) * x_channel_mask).sum(dim=[1,2])
			neg_ldj = -ldj
			
			loss, _, _ = self._calc_loss(neg_ldj, neglog_prob, x_length)

			return loss


	def _eval_batch_discrete(self, x_in, x_length):
		_, ldj = self.model(x_in, reverse=False, length=x_length)
		loss = (-ldj / x_length.float()).mean()
		return loss


	def _calc_loss(self, neg_ldj, neglog_prob, x_length, take_mean=True):
		neg_ldj = (neg_ldj / x_length.float())
		neglog_prob = (neglog_prob / x_length.float())
		loss = neg_ldj + neglog_prob
		if take_mean:
			loss = loss.mean()
			neg_ldj = neg_ldj.mean()
			neglog_prob = neglog_prob.mean()
		return loss, neg_ldj, neglog_prob


	def _preprocess_batch(self, batch):
		x_in = batch
		x_length = x_in.new_zeros(x_in.size(0), dtype=torch.long) + x_in.size(1)
		x_channel_mask = x_in.new_ones(x_in.size(0), x_in.size(1), 1, dtype=torch.float32)
		return x_in, x_length, x_channel_mask


	def initialize(self, num_batches=16):
		if self.model.need_data_init():
			print("Preparing data dependent initialization...")
			batch_list = []
			for _ in range(num_batches):
				batch = self._get_next_batch()
				batch = TaskTemplate.batch_to_device(batch)
				x_in, x_length, _ = self._preprocess_batch(batch)
				batch_tuple = (x_in, {"length": x_length})
				batch_list.append(batch_tuple)
			self.model.initialize_data_dependent(batch_list)