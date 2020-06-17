import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append("../../")

from general.mutils import get_param_val, one_hot, create_transformer_mask, create_channel_mask
from layers.flows.flow_model import FlowModel 
from layers.networks.graph_layers import RGCNNet, RelationGraphAttention


class GraphNodeRNN(FlowModel):

	def __init__(self, model_params, dataset_class, **kwargs):
		super().__init__(layers=None)
		self.model_params = model_params
		self.dataset_class = dataset_class
		self._create_layers()
		self.print_overview()


	##############################
	## Initialization of layers ##
	##############################

	def _create_layers(self):
		# Load global model params
		self.embed_dim = 1
		self.num_node_types = self.dataset_class.num_node_types()

		self.hidden_size = get_param_val(self.model_params, "coupling_hidden_size", default_val=384)
		self.hidden_layers = get_param_val(self.model_params, "coupling_hidden_layers", default_val=5)
		dropout = get_param_val(self.model_params, "coupling_dropout", default_val=0.0)

		self.embed_layer = nn.Embedding(self.num_node_types+1, self.hidden_size)
		self.graph_layer = RGCNNet(c_in=self.hidden_size,
								   c_out=self.num_node_types,
								   num_edges=1,
								   num_layers=self.hidden_layers,
								   hidden_size=self.hidden_size,
								   dp_rate=dropout,
								   rgc_layer_fun=RelationGraphAttention)

	###################
	## RNN Execution ##
	###################

	def _create_labels(self, nodes, length, max_batch_len=None):
		if max_batch_len:
			max_batch_len = length.max()
		torch_range = torch.arange(max_batch_len, device=length.device)
		binary_range = (torch_range[None,:] < length[:,None]).float()
		
		def _mask_labels(samp_lengths):
			nodes_mask = (torch_range[None,:] < samp_lengths).long()
			labels = one_hot(samp_lengths.squeeze(dim=-1), num_classes=max_batch_len).long() * (nodes + 1)
			labels = labels * binary_range.long()
			labels_one_hot = one_hot(labels, num_classes=self.num_node_types+1)[...,1:]
			return nodes_mask, labels_one_hot

		if self.training:
			uniform_probs = binary_range / binary_range.sum(dim=1, keepdims=True)
			sampled_lengths = torch.multinomial(uniform_probs, num_samples=1, replacement=True)
			label_list = [_mask_labels(sampled_lengths)]
		else:
			label_list = [_mask_labels(length.new_zeros(length.size(0),1)+i) for i in range(max_batch_len)]

		return label_list


	def forward(self, z, adjacency=None, ldj=None, reverse=False, length=None, sample_temp=1.0, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0), dtype=torch.float32)
		if length is not None:
			kwargs["length"] = length
			kwargs["src_key_padding_mask"] = create_transformer_mask(length, max_len=z.size(1))
			kwargs["channel_padding_mask"] = create_channel_mask(length, max_len=z.size(1))

		ldj_per_layer = []
		if not reverse:
			orig_nodes = z
			label_list = self._create_labels(z, length, max_batch_len=z.shape[1])
			batch_ldj = z.new_zeros(z.size(0), dtype=torch.float32)
			for nodes_mask, labels_one_hot in label_list:
				z = (orig_nodes + 1) * nodes_mask
				## Run RNN
				z_nodes = self.embed_layer(z)
				out_pred = self.graph_layer(z_nodes, adjacency, **kwargs)
				out_pred = F.log_softmax(out_pred, dim=-1)
				## Calculate loss
				class_ldj = (out_pred * labels_one_hot).sum(dim=-1)
				batch_ldj = batch_ldj + class_ldj.sum(dim=1)
			if len(label_list) > 1:
				batch_ldj = batch_ldj / length.float()
			ldj = ldj + batch_ldj
			return ldj
		else:
			z_nodes = z.new_zeros(z.size(0), z.size(1), dtype=torch.long)
			for rnn_iter in range(length.max()):
				node_embed = self.embed_layer(z_nodes)
				out_pred = self.graph_layer(node_embed, adjacency, **kwargs)
				out_pred = F.log_softmax(out_pred, dim=-1)
				out_pred = out_pred[:,rnn_iter,:]
				if sample_temp > 0.0:
					out_pred = out_pred / sample_temp
					out_pred = torch.softmax(out_pred, dim=-1)
					out_sample = torch.multinomial(out_pred, num_samples=1, replacement=True).squeeze()
				else:
					out_sample = torch.argmax(out_pred, dim=-1)
				z_nodes[:,rnn_iter] = out_sample + 1
			z_nodes = (z_nodes - 1) * kwargs["channel_padding_mask"].squeeze(dim=-1).long()
			return z_nodes, None


	def initialize_data_dependent(self, **kwargs):
		pass

	def need_data_init(self):
		return False

	def test_reversibility(self, *args, **kwargs):
		return True

	def print_overview(self):
		# Retrieve layer descriptions for all flows
		s = "="*30 + "\n"
		s += "GraphNodeRNN network\n"
		s += "-"*30 + "\n"
		s += "-> Hidden size: %i\n" % self.hidden_size
		s += "-> Num layers: %i\n" % self.hidden_layers
		s += "="*30 
		print(s)


if __name__ == '__main__':
	pass
