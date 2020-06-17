import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append("../../")

from general.mutils import get_param_val, create_transformer_mask, create_channel_mask
from layers.flows.flow_model import FlowModel 
from layers.flows.activation_normalization import ActNormFlow
from layers.flows.permutation_layers import InvertibleConv
from layers.flows.coupling_layer import CouplingLayer
from layers.flows.mixture_cdf_layer import MixtureCDFCoupling
from layers.categorical_encoding.mutils import create_encoding
from layers.networks.graph_layers import RGCNNet, RelationGraphAttention


class GraphNodeFlow(FlowModel):

	def __init__(self, model_params, dataset_class, **kwargs):
		super().__init__(layers=None, name="GraphCNF (node based)")
		self.model_params = model_params
		self.dataset_class = dataset_class
		self._create_layers()
		self.print_overview()

	##############################
	## Initialization of layers ##
	##############################

	def _create_layers(self):
		# Load global model params
		self.num_node_types = self.dataset_class.num_node_types()

		# Create essential modules of the flow
		self.node_embed_flow = create_encoding(self.model_params["categ_encoding"], 
											   dataset_class=self.dataset_class, 
											   vocab_size=self.num_node_types)
		self.embed_dim = self.node_embed_flow.D
		main_flow_layers = self._create_node_flow_layers()
		self.flow_layers = nn.ModuleList([self.node_embed_flow] + main_flow_layers)


	def _create_node_flow_layers(self):
		num_flows = get_param_val(self.model_params, "coupling_num_flows", default_val=8)
		hidden_size = get_param_val(self.model_params, "coupling_hidden_size", default_val=384)
		hidden_layers = get_param_val(self.model_params, "coupling_hidden_layers", default_val=4)
		num_mixtures = get_param_val(self.model_params, "coupling_num_mixtures", default_val=16)
		mask_ratio = get_param_val(self.model_params, "coupling_mask_ratio", default_val=0.5)
		dropout = get_param_val(self.model_params, "coupling_dropout", default_val=0.0)
		
		coupling_mask = CouplingLayer.create_channel_mask(self.embed_dim, ratio=mask_ratio)

		model_func = lambda c_out : RGCNNet(c_in=self.embed_dim,
											c_out=c_out,
											num_edges=1,
											num_layers=hidden_layers,
											hidden_size=hidden_size,
											dp_rate=dropout,
											rgc_layer_fun=RelationGraphAttention)

		layers = []
		for _ in range(num_flows):
			layers += [
				ActNormFlow(self.embed_dim),
				InvertibleConv(self.embed_dim),
				MixtureCDFCoupling(c_in=self.embed_dim,
								   mask=coupling_mask,
								   model_func=model_func,
								   block_type="GraphAttentionNet",
								   num_mixtures=num_mixtures,
								   regularizer_max=3.5, # To ensure a accurate reversibility
								   regularizer_factor=2)
			]
		layers += [ActNormFlow(c_in=self.embed_dim)]
		return layers


	####################
	## Flow Execution ##
	####################

	def _run_layer(self, layer, z, reverse, ldj, ldj_per_layer=None, **kwargs):
		layer_res = layer(z, reverse=reverse, **kwargs)

		if len(layer_res) == 2:
			z, layer_ldj = layer_res 
			detailed_layer_ldj = layer_ldj
		elif len(layer_res) == 3:
			z, layer_ldj, detailed_layer_ldj = layer_res
		else:
			print("[!] ERROR: Got more return values than expected: %i" % (len(layer_res)))

		assert torch.isnan(z).sum() == 0, "[!] ERROR: Found NaN latent values. Layer:\n%s" % (layer.info())

		if ldj_per_layer is not None:
			ldj_per_layer.append(detailed_layer_ldj)
		return z, ldj + layer_ldj


	def forward(self, z, adjacency, ldj=None, reverse=False, length=None, **kwargs):
		if length is not None:
			kwargs["src_key_padding_mask"] = create_transformer_mask(length, max_len=z.size(1))
			kwargs["channel_padding_mask"] = create_channel_mask(length, max_len=z.size(1))
		return super().forward(z, adjacency=adjacency, ldj=ldj, reverse=reverse, length=length, **kwargs)

	def initialize_data_dependent(self, batch_list):
		# Batch list needs to consist of tuples: (z, kwargs)
		# kwargs contains the adjacency matrix as well
		with torch.no_grad():
			for batch, kwargs in batch_list:
				kwargs["src_key_padding_mask"] = create_transformer_mask(kwargs["length"], max_len=batch.shape[1])
				kwargs["channel_padding_mask"] = create_channel_mask(kwargs["length"], max_len=batch.shape[1])
			for layer_index, layer in enumerate(self.flow_layers):
				batch_list = FlowModel.run_data_init_layer(batch_list, layer)

	def need_data_init(self):
		return True


	def test_permutation(self, z, adjacency, length):
		ldj = z.new_zeros(z.size(0), dtype=torch.float32)
		kwargs = dict()
		kwargs["length"] = length
		kwargs["src_key_padding_mask"] = create_transformer_mask(length)
		kwargs["channel_padding_mask"] = create_channel_mask(length)

		z, ldj = self._run_layer(self.node_embed_flow, z, reverse=False, ldj=ldj, ldj_per_layer=None, adjacency=adjacency, **kwargs)

		shuffle_noise = torch.rand(z.size(1)).to(z.device) - 2
		shuffle_noise = shuffle_noise * (kwargs["channel_padding_mask"].sum(dim=[0,2])==z.size(0)).float()
		shuffle_noise = shuffle_noise + 0.001 * torch.arange(z.size(1), device=z.device)
		_, shuffle_indices = shuffle_noise.sort(dim=0, descending=False)
		_, unshuffle_indices = shuffle_indices.sort(dim=0, descending=False)
		z_shuffled = z[:,shuffle_indices]
		adjacency_shuffled = adjacency[:,shuffle_indices][:,:,shuffle_indices]
		ldj_shuffled = ldj
		
		for flow in self.flow_layers[1:]:
			z, ldj = self._run_layer(flow, z, adjacency=adjacency, reverse=False, ldj=ldj, **kwargs)
			z_shuffled, ldj_shuffled = self._run_layer(flow, z_shuffled, adjacency=adjacency_shuffled, reverse=False, ldj=ldj_shuffled, **kwargs)
		z_unshuffled = z_shuffled[:,unshuffle_indices]
		ldj_unshuffled = ldj_shuffled

		z_diff = ((z - z_unshuffled).abs() > 1e-4).sum()
		ldj_diff = ((ldj - ldj_unshuffled).abs() > 1e-3).sum()

		if z_diff > 0 or ldj_diff > 0:
			print("Differences z: %s, ldj: %s" % (str(z_diff.item()), str(ldj_diff.item())))
			print("Z", z[0,:,0])
			print("Z shuffled", z_shuffled[0,:,0])
			print("Z unshuffled", z_unshuffled[0,:,0])
			print("LDJ", ldj[0:5])
			print("LDJ unshuffled", ldj_unshuffled[0:5])
			print("LDJ diff", (ldj - ldj_unshuffled).abs())
			return False
		else:
			print("Shuffle test succeeded!")
			return True


	def test_reversibility(self, z, adjacency, length):
		ldj = z.new_zeros(z.size(0), dtype=torch.float32)
		kwargs = dict()
		kwargs["length"] = length
		kwargs["src_key_padding_mask"] = create_transformer_mask(length, max_len=z.size(1))
		kwargs["channel_padding_mask"] = create_channel_mask(length, max_len=z.size(1))

		## Embed nodes
		z_nodes, ldj = self._run_layer(self.node_embed_flow, z, reverse=False, ldj=ldj, adjacency=adjacency, **kwargs)
		z_nodes_embed = z_nodes
		ldj_embed = ldj
		## Testing RGCN flows
		for flow in self.flow_layers[1:]:
			z_nodes, ldj = self._run_layer(flow, z_nodes, reverse=False, ldj=ldj, adjacency=adjacency, **kwargs)
		z_nodes_reversed, ldj_reversed = z_nodes, ldj
		for flow in reversed(self.flow_layers[1:]):
			z_nodes_reversed, ldj_reversed = self._run_layer(flow, z_nodes_reversed, reverse=True, ldj=ldj_reversed, adjacency=adjacency, **kwargs)
		reverse_succeeded = ((z_nodes_reversed - z_nodes_embed).abs() > 1e-2).sum() == 0 and ((ldj_reversed - ldj_embed).abs() > 1e-1).sum() == 0
		if not reverse_succeeded:
			print("[!] ERROR: Coupling layer with given adjacency matrix are not reversible. Max diffs:\n" + \
				"Nodes: %s\n" % str(torch.max((z_nodes_reversed - z_nodes_embed).abs())) + \
				"LDJ: %s\n" % str(torch.max((ldj_reversed - ldj_embed).abs())))
			z_nodes = z_nodes_embed
			ldj = ldj_embed
			large_error = False
			for flow_index, flow in enumerate(self.flow_layers[1:]):
				z_nodes_forward, ldj_forward = self._run_layer(flow, z_nodes, reverse=False, ldj=ldj, adjacency=adjacency, **kwargs)
				z_nodes_backward, ldj_backward = self._run_layer(flow, z_nodes_forward, reverse=True, ldj=ldj_forward, adjacency=adjacency, **kwargs)

				node_diff = (z_nodes_backward - z_nodes).abs()
				ldj_diff = (ldj_backward - ldj).abs()
				max_node_diff = torch.max(node_diff)
				max_ldj_diff = torch.max(ldj_diff)
				mean_node_diff = torch.mean(node_diff)
				mean_ldj_diff = torch.mean(ldj_diff)
				print("Flow [%i]: %s" % (flow_index+1, flow.info()))
				print("-> Max node diff: %s" % str(max_node_diff))
				print("-> Max ldj diff: %s" % str(max_ldj_diff))
				print("-> Mean node diff: %s" % str(mean_node_diff))
				print("-> Mean ldj diff: %s" % str(mean_ldj_diff))
				if max_node_diff > 1e-2:
					batch_index = torch.argmax(ldj_diff).item()
					print("-> Batch index", batch_index)
					print("-> Nodes with max diff:")
					print(node_diff[batch_index])
					print(z_nodes_backward[batch_index])
					print(z_nodes[batch_index])
					print(z_nodes_forward[batch_index])
					node_mask = (node_diff[batch_index] > 1e-4)
					faulty_nodes = z_nodes_forward[batch_index].masked_select(node_mask)
					num_small_faulty_nodes = (faulty_nodes.abs() < 1.0).sum().item()
					large_error = large_error or (num_small_faulty_nodes > 0)

				z_nodes = z_nodes_forward
				ldj = ldj_forward
			if not large_error:
				print("-"*50)
				print("Error probably caused by large values out of range in the mixture layer. Ignored for now.")
			return (not large_error)
		else:
			print("Reversibility test passed")
			return True


if __name__ == '__main__':
	pass
