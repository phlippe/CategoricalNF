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
from layers.flows.distributions import create_prior_distribution

from layers.categorical_encoding.mutils import create_encoding
from layers.categorical_encoding.linear_encoding import LinearCategoricalEncoding
from layers.categorical_encoding.decoder import DecoderLinear

from layers.networks.graph_layers import *
from experiments.molecule_generation.graph_node_edge_coupling import *
from experiments.molecule_generation.mutils import adjacency2pairs, get_adjacency_indices, pairs2adjacency



class GraphCNF(FlowModel):

	def __init__(self, model_params, dataset_class, **kwargs):
		super().__init__(layers=None, name="GraphCNF")
		self.model_params = model_params
		self.dataset_class = dataset_class
		self._create_layers()
		self.print_overview()

	##############################
	## Initialization of layers ##
	##############################

	def _create_layers(self):
		# Load global model params
		self.max_num_nodes = self.dataset_class.max_num_nodes()
		self.num_node_types = self.dataset_class.num_node_types()
		self.num_edge_types = self.dataset_class.num_edge_types()
		self.num_max_neighbours = self.dataset_class.num_max_neighbours()
		# Prior distribution is needed here for edges
		prior_config = get_param_val(self.model_params, "prior_distribution", default_val=dict())
		self.prior_distribution = create_prior_distribution(prior_config)
		# Create encoding and flow layers
		self._create_encoding_layers()
		self._create_step_flows()


	def _create_encoding_layers(self):
		self.node_encoding = create_encoding(self.model_params["categ_encoding_nodes"], 
											 dataset_class=self.dataset_class, 
											 vocab_size=self.num_node_types,
											 category_prior=self.dataset_class.get_node_prior(data_root="data/"))
		self.edge_attr_encoding = create_encoding(self.model_params["categ_encoding_edges"], 
												  dataset_class=self.dataset_class, 
												  vocab_size=self.num_edge_types, # Removing the virtual edges here
												  category_prior=self.dataset_class.get_edge_prior(data_root="data/"))

		self.encoding_dim_nodes = self.node_encoding.D
		self.encoding_dim_edges = self.edge_attr_encoding.D

		# Virtual edges are encoded by a single mixture
		self.edge_virtual_encoding = LinearCategoricalEncoding(num_dimensions=self.encoding_dim_edges,
															   flow_config={"num_flows": self.model_params["encoding_virtual_num_flows"], 
															   				"hidden_layers": 2, 
															   				"hidden_size": 128},
															   dataset_class=self.dataset_class,
															   vocab_size=1)
		# Posterior needs to be a separate network as the true cannot be easily found. 
		self.edge_virtual_decoder = DecoderLinear(num_categories=2, embed_dim=self.encoding_dim_edges, 
												  hidden_size=128, num_layers=2, 
												  class_prior_log=np.log(np.array([0.9, 0.1]))) # Molecules are sparse and usually have ~10% density

	def _create_step_flows(self):
		## Get hyperparameters from model_params dictionary
		hidden_size_nodes = get_param_val(self.model_params, "coupling_hidden_size_nodes", default_val=256)
		hidden_size_edges = get_param_val(self.model_params, "coupling_hidden_size_edges", default_val=128)
		num_flows = get_param_val(self.model_params, "coupling_num_flows", default_val="4,6,6")
		num_flows = [int(k) for k in num_flows.split(",")]
		hidden_layers = get_param_val(self.model_params, "coupling_hidden_layers", default_val=4)
		if isinstance(hidden_layers, str):
			if "," in hidden_layers:
				hidden_layers = [int(l) for l in hidden_layers.split(",")]
			else:
				hidden_layers = [int(hidden_layers)]*3
		else:
			hidden_layers = [hidden_layers]*3
		num_mixtures_nodes = get_param_val(self.model_params, "coupling_num_mixtures_nodes", default_val=16)
		num_mixtures_edges = get_param_val(self.model_params, "coupling_num_mixtures_edges", default_val=16)
		mask_ratio = get_param_val(self.model_params, "coupling_mask_ratio", default_val=0.5)
		dropout = get_param_val(self.model_params, "coupling_dropout", default_val=0.0)

		#----------------#
		#- Step 1 flows -#
		#----------------#

		coupling_mask_nodes = CouplingLayer.create_channel_mask(self.encoding_dim_nodes, ratio=mask_ratio)
		step1_model_func = lambda c_out : RGCNNet(c_in=self.encoding_dim_nodes,
													c_out=c_out,
													num_edges=self.num_edge_types,
													num_layers=hidden_layers[0],
													hidden_size=hidden_size_nodes,
													max_neighbours=self.dataset_class.num_max_neighbours(),
													dp_rate=dropout,
													rgc_layer_fun=RelationGraphConv)
		step1_flows = []
		for _ in range(num_flows[0]):
			step1_flows += [
				ActNormFlow(self.encoding_dim_nodes),
				InvertibleConv(self.encoding_dim_nodes),
				MixtureCDFCoupling(c_in=self.encoding_dim_nodes,
								   mask=coupling_mask_nodes,
								   model_func=step1_model_func,
								   block_type="RelationGraphConv",
								   num_mixtures=num_mixtures_nodes,
								   regularizer_max=3.5, # To ensure a accurate reversibility
								   regularizer_factor=2)
			]
		self.step1_flows = nn.ModuleList(step1_flows)

		#------------------#
		#- Step 2+3 flows -#
		#------------------#

		coupling_mask_edges = CouplingLayer.create_channel_mask(self.encoding_dim_edges, ratio=mask_ratio)

		# Definition of the Edge-GNN network
		def edge2node_layer_func(step_idx):
			if step_idx == 1:
				return lambda : Edge2NodeAttnLayer(hidden_size_nodes=hidden_size_nodes, 
												   hidden_size_edges=hidden_size_edges,
												   skip_config=2)
			else:
				return lambda : Edge2NodeQKVAttnLayer(hidden_size_nodes=hidden_size_nodes, 
													  hidden_size_edges=hidden_size_edges,
													  skip_config=2)

		node2edge_layer_func = lambda : Node2EdgePlainLayer(hidden_size_nodes=hidden_size_nodes, 
															hidden_size_edges=hidden_size_edges,
															skip_config=2)
		def edge_gnn_layer_func(step_idx):
			return lambda : EdgeGNNLayer(edge2node_layer_func=edge2node_layer_func(step_idx), 
										 node2edge_layer_func=node2edge_layer_func)
		def get_model_func(step_idx):
			return lambda c_out_nodes, c_out_edges :  EdgeGNN(c_in_nodes=self.encoding_dim_nodes,
															  c_in_edges=self.encoding_dim_edges,
															  c_out_nodes=c_out_nodes,
															  c_out_edges=c_out_edges,
															  edge_gnn_layer_func=edge_gnn_layer_func(step_idx),
															  max_neighbours=self.dataset_class.num_max_neighbours(),
															  num_layers=hidden_layers[step_idx])

		# Activation normalization and invertible 1x1 convolution need to be applied on both nodes and edges independently.
		# The "NodeEdgeFlowWrapper" handles the forward pass for such flows
		actnorm_layer = lambda : NodeEdgeFlowWrapper(node_flow=ActNormFlow(c_in=self.encoding_dim_nodes),
													 edge_flow=ActNormFlow(c_in=self.encoding_dim_edges))
		permut_layer = lambda : NodeEdgeFlowWrapper(node_flow=InvertibleConv(c_in=self.encoding_dim_nodes),
													edge_flow=InvertibleConv(c_in=self.encoding_dim_edges))
		coupling_layer = lambda step_idx : NodeEdgeCoupling(c_in_nodes=self.encoding_dim_nodes,
														   c_in_edges=self.encoding_dim_edges,
														   mask_nodes=coupling_mask_nodes,
														   mask_edges=coupling_mask_edges,
														   num_mixtures_nodes=num_mixtures_nodes,
														   num_mixtures_edges=num_mixtures_edges,
														   model_func=get_model_func(step_idx),
														   regularizer_max=3.5, # To ensure a accurate reversibility
														   regularizer_factor=2)
		

		step2_flows = []
		for _ in range(num_flows[1]):
			step2_flows += [
				actnorm_layer(),
				permut_layer(),
				coupling_layer(step_idx=1)
			]
		self.step2_flows = nn.ModuleList(step2_flows)

		step3_flows = []
		for _ in range(num_flows[2]):
			step3_flows += [
				actnorm_layer(),
				permut_layer(),
				coupling_layer(step_idx=2)
			]
		self.step3_flows = nn.ModuleList(step3_flows)



	####################
	## Flow Execution ##
	####################

	def _run_layer(self, layer, z, reverse, ldj, ldj_per_layer=None, **kwargs):
		## Function to reduce output handling in main forward pass
		layer_res = layer(z, reverse=reverse, **kwargs)
		if len(layer_res) == 2:
			z, layer_ldj = layer_res 
			detailed_layer_ldj = layer_ldj
		elif len(layer_res) == 3:
			z, layer_ldj, detailed_layer_ldj = layer_res
		if ldj_per_layer is not None:
			ldj_per_layer.append(detailed_layer_ldj)
		return z, ldj + layer_ldj

	def _run_node_edge_layer(self, layer, z_nodes, z_edges, reverse, ldj, ldj_per_layer=None, **kwargs):
		## Function to reduce output handling in main forward pass
		layer_res = layer(z_nodes=z_nodes, z_edges=z_edges, reverse=reverse, **kwargs)
		if len(layer_res) == 3:
			z_nodes, z_edges, layer_ldj = layer_res
			detailed_layer_ldj = layer_ldj
		elif len(layer_res) == 4:
			z_nodes, z_edges, layer_ldj, detailed_layer_ldj = layer_res
		if ldj_per_layer is not None:
			ldj_per_layer.append(detailed_layer_ldj)
		return z_nodes, z_edges, ldj + layer_ldj 

	def forward(self, z, adjacency=None, ldj=None, reverse=False, get_ldj_per_layer=False, length=None, sample_temp=1.0, **kwargs):
		z_nodes = z # Renaming as argument "z" is usually used for the flows, but here it represents the discrete node types
		if ldj is None:
			ldj = z_nodes.new_zeros(z_nodes.size(0), dtype=torch.float32)
		if length is not None:
			kwargs["length"] = length
			kwargs["channel_padding_mask"] = create_channel_mask(length, max_len=z_nodes.size(1))

		ldj_per_layer = []
		if not reverse:
			## Step 1 => Encode nodes in latent space and apply RGCN flows
			z_nodes, ldj = self._step1_forward(z_nodes, adjacency, ldj, False, ldj_per_layer, **kwargs)
			## Edges are represented as a list (1D tensor), not as a matrix, because we do not want to consider each edge twice
			# X_indices is a tuple, where each element has the size of the edge tensor and states the node indices of the corresponding edge
			# Mask_valid is a tensor of the same size, but contains 0 for those edges that are not "valid".
			# This is when edges are padding elements for graphs of different sizes 
			z_edges_disc, x_indices, mask_valid = adjacency2pairs(adjacency=adjacency, length=length)
			kwargs["mask_valid"] = mask_valid * (z_edges_disc != 0).to(mask_valid.dtype)
			kwargs["x_indices"] = x_indices
			binary_adjacency = (adjacency > 0).long()
			## Step 2 => Encode edge attributes in latent space and apply first EdgeGNN flows
			z_nodes, z_edges, ldj = self._step2_forward(z_nodes, z_edges_disc, ldj, False, ldj_per_layer, 
														binary_adjacency=binary_adjacency, **kwargs)
			
			## Step 3 => Encode virtual edges in latent space and apply final EdgeGNN flows
			kwargs["mask_valid"] = mask_valid
			virtual_edge_mask = mask_valid * (z_edges_disc == 0).float()
			z_nodes, z_edges, ldj = self._step3_forward(z_nodes, z_edges, ldj, False, ldj_per_layer, virtual_edge_mask, **kwargs)

			## Add log probability of adjacency matrix to ldj. Only nodes are considered in the task object
			adjacency_log_prob = (self.prior_distribution.log_prob(z_edges) * mask_valid.unsqueeze(dim=-1)).sum(dim=[1,2])
			ldj = ldj + adjacency_log_prob
			ldj_per_layer.append({"adjacency_log_prob": adjacency_log_prob})
		else:
			z_nodes = z
			batch_size, num_nodes = z_nodes.size(0), z_nodes.size(1)
			## Sample latent variables for adjacency matrix
			mask_valid, x_indices = get_adjacency_indices(num_nodes=num_nodes, length=length)
			kwargs["mask_valid"] = mask_valid
			kwargs["x_indices"] = x_indices
			z_edges = self.prior_distribution.sample(shape=(batch_size, mask_valid.size(1), self.encoding_dim_edges),
													 temp=sample_temp).to(z.device)
			## Reverse step 3 => decode virtual edges
			z_nodes, z_edges, ldj, mask_valid = self._step3_forward(z_nodes, z_edges, ldj, True, ldj_per_layer, **kwargs)
			binary_adjacency = pairs2adjacency(num_nodes=num_nodes, pairs=mask_valid, length=length, x_indices=x_indices)
			## Reverse step 2 => decode edge attributes
			kwargs["mask_valid"] = mask_valid
			z_nodes, z_edges, ldj = self._step2_forward(z_nodes, z_edges, ldj, True, ldj_per_layer, 
														binary_adjacency=binary_adjacency, **kwargs)
			adjacency = pairs2adjacency(num_nodes=num_nodes, pairs=z_edges, length=length, x_indices=x_indices)
			## Reverse step 1 => decode node types
			z_nodes, ldj = self._step1_forward(z_nodes, adjacency, ldj, reverse=True, ldj_per_layer=ldj_per_layer, **kwargs)
			z_nodes = (z_nodes, adjacency)

		if get_ldj_per_layer:
			return z_nodes, ldj, ldj_per_layer
		else:
			return z_nodes, ldj

	def _step1_forward(self, z_nodes, adjacency, ldj, reverse, ldj_per_layer, **kwargs):
		if not reverse:
			## Encode node types
			z_nodes, ldj = self._run_layer(self.node_encoding, z_nodes, reverse, ldj=ldj, ldj_per_layer=ldj_per_layer, **kwargs)
			## Run first RGCN coupling layers with full adjacency matrix
			for flow in self.step1_flows:
				z_nodes, ldj = self._run_layer(flow, z_nodes, reverse, ldj=ldj, ldj_per_layer=ldj_per_layer, adjacency=adjacency, **kwargs)
		else:
			## Reverse RGCN coupling layers with full adjacency matrix
			for flow in reversed(self.step1_flows):
				z_nodes, ldj = self._run_layer(flow, z_nodes, reverse, ldj=ldj, ldj_per_layer=ldj_per_layer, adjacency=adjacency, **kwargs)
			## Reverse embedding of nodes
			z_nodes, ldj = self._run_layer(self.node_encoding, z_nodes, reverse, ldj=ldj, ldj_per_layer=ldj_per_layer, **kwargs)
		return z_nodes, ldj

	def _step2_forward(self, z_nodes, z_edges, ldj, reverse, ldj_per_layer, **kwargs):
		if not reverse:
			## Encode edge attributes
			kwargs_edge_embed = kwargs.copy()
			kwargs_edge_embed["channel_padding_mask"] = kwargs["mask_valid"].unsqueeze(dim=-1)
			z_attr = (z_edges-1).clamp(min=0)
			z_edges, ldj = self._run_layer(self.edge_attr_encoding, z_attr, reverse, ldj, ldj_per_layer, **kwargs_edge_embed)
			## Running node-edge coupling layers
			for flow in self.step2_flows:
				z_nodes, z_edges, ldj = self._run_node_edge_layer(flow, z_nodes, z_edges, reverse, ldj, ldj_per_layer, **kwargs)
		else:
			## Reverse edge attribute layers
			for flow in reversed(self.step2_flows):
				z_nodes, z_edges, ldj = self._run_node_edge_layer(flow, z_nodes, z_edges, reverse, ldj, ldj_per_layer, **kwargs)
			## Reverse adjacency matrix embedding
			kwargs_edge_embed = kwargs.copy()
			kwargs_edge_embed["channel_padding_mask"] = kwargs["mask_valid"].unsqueeze(dim=-1)
			z_edges, ldj = self._run_layer(self.edge_attr_encoding, z_edges, reverse, ldj, ldj_per_layer, **kwargs_edge_embed)
			z_edges = (z_edges+1) * kwargs["mask_valid"].long() # Set masked elements to zero => no edge

		return z_nodes, z_edges, ldj

	def _step3_forward(self, z_nodes, z_edges, ldj, reverse, ldj_per_layer, virtual_edge_mask=None, **kwargs):
		if not reverse:
			## Encode virtual edges
			kwargs_no_edge_embed = kwargs.copy()
			kwargs_no_edge_embed["channel_padding_mask"] = virtual_edge_mask.unsqueeze(dim=-1)
			virt_edges = z_edges.new_zeros(z_edges.shape[:-1], dtype=torch.long)
			z_virtual_edges, ldj = self._run_layer(self.edge_virtual_encoding, virt_edges, reverse, ldj, ldj_per_layer, **kwargs_no_edge_embed)
			z_edges = torch.where(virtual_edge_mask.unsqueeze(dim=-1)==1, z_virtual_edges, z_edges)

			## Run decoder
			edge_log_probs = self.edge_virtual_decoder(z_edges)
			edge_ldj = torch.where(virtual_edge_mask == 1, edge_log_probs[...,0], edge_log_probs[...,1] * kwargs["mask_valid"]).sum(dim=-1)
			ldj = ldj + edge_ldj * (kwargs["beta"] if "beta" in kwargs else 1.0)
			# Debug information
			with torch.no_grad():
				avg_edge_ldj = edge_ldj / kwargs["mask_valid"].sum(dim=-1)
				ldj_per_layer.append({"virtual_edges_bpd": np.log2(np.exp(1))*avg_edge_ldj})
			
			## Running node-edge coupling layers
			for flow in self.step3_flows:
				z_nodes, z_edges, ldj = self._run_node_edge_layer(flow, z_nodes, z_edges, reverse, ldj, ldj_per_layer, **kwargs)
			return z_nodes, z_edges, ldj
		else:
			## Reverse node-edge coupling layers
			for flow in reversed(self.step3_flows):
				z_nodes, z_edges, ldj = self._run_node_edge_layer(flow, z_nodes, z_edges, reverse, ldj, ldj_per_layer, **kwargs)
			## Determine virtual edges
			is_edge = self.edge_virtual_decoder(z_edges).argmax(dim=-1)
			mask_valid = kwargs["mask_valid"] * (is_edge == 1).float()
			return z_nodes, z_edges, ldj, mask_valid


	def initialize_data_dependent(self, batch_list):
		# Batch list needs to consist of tuples: (z, kwargs)
		# kwargs contains the adjacency matrix as well
		with torch.no_grad():
			
			for batch, kwargs in batch_list:
				kwargs["channel_padding_mask"] = create_channel_mask(kwargs["length"], max_len=batch.shape[1])

			for module_index, module_list in enumerate([[self.node_encoding], self.step1_flows]):
				for layer_index, layer in enumerate(module_list):
					print("Processing layer %i (module %i)..." % (layer_index+1, module_index+1), end="\r")
					if isinstance(layer, FlowLayer):
						batch_list = FlowModel.run_data_init_layer(batch_list, layer)
					elif isinstance(layer, FlowModel):
						batch_list = layer.initialize_data_dependent(batch_list)
					else:
						print("[!] ERROR: Unknown layer type", layer)
						sys.exit(1)

			## Initialize main flow
			for i in range(len(batch_list)):
				z_nodes, kwargs = batch_list[i]
				z_adjacency, x_indices, mask_valid = adjacency2pairs(adjacency=kwargs["adjacency"], length=kwargs["length"])
				attr_mask_valid = mask_valid * (z_adjacency != 0).to(mask_valid.dtype)
				z_edges, _, _ = self.edge_attr_encoding((z_adjacency-1).clamp(min=0), 
														reverse=False, 
														channel_padding_mask=attr_mask_valid.unsqueeze(dim=-1))
				kwargs["original_z_adjacency"] = z_adjacency
				kwargs["binary_adjacency"] = (kwargs["adjacency"] > 0).long()
				kwargs["original_mask_valid"] = mask_valid
				kwargs["mask_valid"] = attr_mask_valid
				kwargs["x_indices"] = x_indices
				batch_list[i] = ([z_nodes, z_edges], kwargs)

			for layer_index, layer in enumerate(self.step2_flows):
				batch_list = FlowModel.run_data_init_layer(batch_list, layer)

			for i in range(len(batch_list)):
				z, kwargs = batch_list[i]
				z_nodes, z_edges = z[0], z[1]
				no_edge_mask_valid = kwargs["original_mask_valid"] * (kwargs["original_z_adjacency"] == 0).float()
				z_no_edges, _, _ = self.edge_virtual_encoding(torch.zeros_like(kwargs["original_z_adjacency"]), 
															  reverse=False, 
															  channel_padding_mask=no_edge_mask_valid.unsqueeze(dim=-1))
				z_edges = z_edges * (1 - no_edge_mask_valid)[...,None] + z_no_edges * no_edge_mask_valid[...,None]
				kwargs["mask_valid"] = kwargs["original_mask_valid"]
				kwargs.pop("binary_adjacency")
				batch_list[i] = ([z_nodes, z_edges], kwargs)
			
			for layer_index, layer in enumerate(self.step3_flows):
				batch_list = FlowModel.run_data_init_layer(batch_list, layer)


	def need_data_init(self):
		return True


	def test_reversibility(self, z_nodes, adjacency, length, **kwargs):
		ldj = z_nodes.new_zeros(z_nodes.size(0), dtype=torch.float32)
		if length is not None:
			kwargs["length"] = length
			kwargs["channel_padding_mask"] = create_channel_mask(length, max_len=z_nodes.size(1))

		## Performing encoding of step 1
		z_nodes, ldj = self._run_layer(self.node_encoding, z_nodes, False, ldj=ldj, **kwargs)
		z_nodes_embed = z_nodes
		ldj_embed = ldj

		## Testing step 1 flows
		for flow in self.step1_flows:
			z_nodes, ldj = self._run_layer(flow, z_nodes, reverse=False, ldj=ldj, adjacency=adjacency, **kwargs)
		z_nodes_reversed, ldj_reversed = z_nodes, ldj
		for flow in reversed(self.step1_flows):
			z_nodes_reversed, ldj_reversed = self._run_layer(flow, z_nodes_reversed, reverse=True, ldj=ldj_reversed, adjacency=adjacency, **kwargs)
		rev_node = ((z_nodes_reversed - z_nodes_embed).abs() > 1e-3).sum() == 0 and ((ldj_reversed - ldj_embed).abs() > 1e-1).sum() == 0
		if not rev_node:
			print("[#] WARNING: Step 1 - Coupling layers are not precisely reversible. Max diffs:\n" + \
					"Nodes: %s\n" % str(torch.max((z_nodes_reversed - z_nodes_embed).abs())) + \
					"LDJ: %s" % str(torch.max((ldj_reversed - ldj_embed).abs())))
		
		## Performing encoding of step 2
		z_edges_disc, x_indices, mask_valid = adjacency2pairs(adjacency=adjacency, length=length)
		kwargs["mask_valid"] = mask_valid * (z_edges_disc != 0).to(mask_valid.dtype)
		kwargs["x_indices"] = x_indices
		binary_adjacency = (adjacency > 0).long()
		kwargs_edge_embed = kwargs.copy()
		kwargs_edge_embed["channel_padding_mask"] = kwargs["mask_valid"].unsqueeze(dim=-1)
		z_attr = (z_edges_disc-1).clamp(min=0)
		z_edges, ldj = self._run_layer(self.edge_attr_encoding, z_attr, False, ldj, **kwargs_edge_embed)

		## Testing step 2 flows
		z_nodes_orig, z_edges_orig, ldj_orig = z_nodes, z_edges, ldj
		for flow in self.step2_flows:
			z_nodes, z_edges, ldj = self._run_node_edge_layer(flow, z_nodes, z_edges, False, ldj, 
															  binary_adjacency=binary_adjacency,**kwargs)
		z_nodes_rev, z_edges_rev, ldj_rev = z_nodes, z_edges, ldj
		for flow in reversed(self.step2_flows):
			z_nodes_rev, z_edges_rev, ldj_rev = self._run_node_edge_layer(flow, z_nodes_rev, z_edges_rev, True, ldj_rev, 
																		  binary_adjacency=binary_adjacency,**kwargs)
		rev_edge_attr = ((z_nodes_rev - z_nodes_orig).abs() > 1e-3).sum() == 0 and \
						((z_edges_rev - z_edges_orig).abs() > 1e-3).sum() == 0 and \
						((ldj_rev - ldj_orig).abs() > 1e-1).sum() == 0
		if not rev_edge_attr:
			print("[#] WARNING: Step 2 - Coupling layers are not precisely reversible. Max diffs:\n" + \
					"Nodes: %s\n" % str(torch.max((z_nodes_rev - z_nodes_orig).abs())) + \
					"Edges: %s\n" % str(torch.max((z_edges_rev - z_edges_orig).abs())) + \
					"LDJ: %s" % str(torch.max((ldj_rev - ldj_orig).abs())))

		## Performing encoding of step 3
		kwargs["mask_valid"] = mask_valid
		virtual_edge_mask = mask_valid * (z_edges_disc == 0).float()
		kwargs_no_edge_embed = kwargs.copy()
		kwargs_no_edge_embed["channel_padding_mask"] = virtual_edge_mask.unsqueeze(dim=-1)
		virt_edges = z_edges.new_zeros(z_edges.shape[:-1], dtype=torch.long)
		z_virtual_edges, ldj = self._run_layer(self.edge_virtual_encoding, virt_edges, False, ldj, **kwargs_no_edge_embed)
		z_edges = torch.where(virtual_edge_mask.unsqueeze(dim=-1)==1, z_virtual_edges, z_edges)

		## Testing step 3 flows
		z_nodes_orig, z_edges_orig, ldj_orig = z_nodes, z_edges, ldj
		for flow in self.step3_flows:
			z_nodes, z_edges, ldj = self._run_node_edge_layer(flow, z_nodes, z_edges, False, ldj, 
															  **kwargs)
		z_nodes_rev, z_edges_rev, ldj_rev = z_nodes, z_edges, ldj
		for flow in reversed(self.step3_flows):
			z_nodes_rev, z_edges_rev, ldj_rev = self._run_node_edge_layer(flow, z_nodes_rev, z_edges_rev, True, ldj_rev, 
																		  **kwargs)
		rev_edge_virt = ((z_nodes_rev - z_nodes_orig).abs() > 1e-3).sum() == 0 and \
						((z_edges_rev - z_edges_orig).abs() > 1e-3).sum() == 0 and \
						((ldj_rev - ldj_orig).abs() > 1e-1).sum() == 0
		if not rev_edge_virt:
			print("[#] WARNING: Step 3 - Coupling layers are not precisely reversible. Max diffs:\n" + \
					"Nodes: %s\n" % str(torch.max((z_nodes_rev - z_nodes_orig).abs())) + \
					"Edges: %s\n" % str(torch.max((z_edges_rev - z_edges_orig).abs())) + \
					"LDJ: %s" % str(torch.max((ldj_rev - ldj_orig).abs())))

		if rev_node and rev_edge_attr and rev_edge_virt:
			print("Reversibility test succeeded!")
		else:
			print("Reversibility test finished with warnings. Non-reversibility can be due to limited precision in mixture coupling layers")


	def print_overview(self):
		# Retrieve layer descriptions for all flows
		layer_descp = list()
		layer_descp.append("(%i) Node %s" % (1, self.node_encoding.info()))
		index_bias = 2
		for layer_index, layer in enumerate(self.step1_flows):
			layer_descp.append("(%i) [Step 1] %s" % (layer_index+index_bias, layer.info()))
		index_bias += len(self.step1_flows)
		layer_descp.append("(%i) Edge attribute %s" % (index_bias, self.edge_attr_encoding.info()))
		index_bias += 1
		for layer_index, layer in enumerate(self.step2_flows):
			layer_descp.append("(%i) [Step 2] %s" % (layer_index+index_bias, layer.info().replace("\n","\n\t      ")))
		index_bias += len(self.step2_flows)
		layer_descp.append("(%i) Virtual Edge %s" % (index_bias, self.edge_virtual_encoding.info()))
		index_bias += 1
		for layer_index, layer in enumerate(self.step3_flows):
			layer_descp.append("(%i) [Step 3] %s" % (layer_index+index_bias, layer.info().replace("\n","\n\t      ")))
		num_tokens = max([20] + [len(s) for s in "\n".join(layer_descp).split("\n")])
		# Print out info in a nicer format
		print("="*num_tokens)
		print("GraphCNF")
		print("-"*num_tokens)
		print("\n".join(layer_descp))
		print("="*num_tokens)


if __name__ == '__main__':
	
	from experiments.molecule_generation.datasets.zinc250k import Zinc250kDataset
	
	model_params = {}
	dataset_class = Zinc250kDataset
	flow = GraphCNF(model_params, dataset_class)

