import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append("../../")

from layers.flows.flow_layer import FlowLayer 
from layers.flows.mixture_cdf_layer import MixtureCDFCoupling


class NodeEdgeCoupling(FlowLayer):
	"""
	Flow layer to apply a logistic mixture coupling on both nodes and edge latent variables at the same time.
	"""


	def __init__(self, c_in_nodes, c_in_edges, 
					   mask_nodes, mask_edges, 
					   num_mixtures_nodes, num_mixtures_edges,
					   model_func, 
					   regularizer_max=-1,
					   regularizer_factor=1,
					   **kwargs):
		super().__init__()
		self.c_in_nodes = c_in_nodes
		self.c_in_edges = c_in_edges
		self.num_mixtures_nodes = num_mixtures_nodes
		self.num_mixtures_edges = num_mixtures_edges
		self.regularizer_max = regularizer_max
		self.regularizer_factor = regularizer_factor
		self.register_buffer("mask_nodes", mask_nodes)
		self.register_buffer("mask_edges", mask_edges)
		self.c_out_nodes = self.c_in_nodes * (2 + 3 * self.num_mixtures_nodes)
		self.c_out_edges = self.c_in_edges * (2 + 3 * self.num_mixtures_edges)

		self.nn = model_func(c_out_nodes=self.c_out_nodes, c_out_edges=self.c_out_edges)

		self.scaling_factor_nodes = nn.Parameter(torch.zeros(self.c_in_nodes))
		self.scaling_factor_edges = nn.Parameter(torch.zeros(self.c_in_edges))
		self.mixture_scaling_factor_nodes = nn.Parameter(torch.zeros(self.c_in_nodes, self.num_mixtures_nodes))
		self.mixture_scaling_factor_edges = nn.Parameter(torch.zeros(self.c_in_edges, self.num_mixtures_edges))


	def forward(self, z_nodes, z_edges, ldj=None, reverse=False, 
					  length=None, channel_padding_mask=None,
					  mask_valid=None, x_indices=None, 
					  binary_adjacency=None, **kwargs):
		if ldj is None:
			ldj = z_nodes.new_zeros(z_nodes.size(0),)
		## Prepare inputs for both nodes and edges
		orig_z_nodes, orig_z_edges = z_nodes, z_edges
		mask_nodes = self.mask_nodes[None,:min(self.mask_nodes.size(0),z_nodes.size(1)),:]
		mask_edges = self.mask_edges[None,:min(self.mask_edges.size(0),z_edges.size(1)),:]
		z_nodes_in = mask_nodes * z_nodes
		z_edges_in = mask_edges * z_edges
		
		## Run EdgeGNN
		nn_nodes_out, nn_edges_out = self.nn(z_nodes=z_nodes_in, z_edges=z_edges_in, 
											 length=length, channel_padding_mask=channel_padding_mask,
											 x_indices=x_indices, mask_valid=mask_valid,
											 binary_adjacency=binary_adjacency
											 )

		## Transform node features
		nn_nodes_out = nn_nodes_out * channel_padding_mask
		z_nodes_out, nodes_ldj, nodes_reg_ldj = self._run_mixture_layer(orig_z=orig_z_nodes,
																		nn_out=nn_nodes_out,
																		mask=mask_nodes,
																		num_mixtures=self.num_mixtures_nodes,
																		scaling_factor=self.scaling_factor_nodes,
																		mixture_scaling_factor=self.mixture_scaling_factor_nodes,
																		channel_padding_mask=channel_padding_mask,
																		reverse=reverse)
		z_nodes_out = z_nodes_out * channel_padding_mask

		## Transform edge features
		mask_valid = mask_valid.unsqueeze(dim=-1)
		nn_edges_out = nn_edges_out * mask_valid
		z_edges_out, edges_ldj, edges_reg_ldj = self._run_mixture_layer(orig_z=orig_z_edges,
																		nn_out=nn_edges_out,
																		mask=mask_edges,
																		num_mixtures=self.num_mixtures_edges,
																		scaling_factor=self.scaling_factor_edges,
																		mixture_scaling_factor=self.mixture_scaling_factor_edges,
																		channel_padding_mask=mask_valid,
																		reverse=reverse)
		z_edges_out = z_edges_out * mask_valid

		## Finalizing coupling layer
		ldj = ldj + nodes_ldj + edges_ldj

		assert torch.isnan(z_nodes_out).sum() == 0 and \
			   torch.isnan(z_edges_out).sum() == 0 and \
			   torch.isnan(ldj).sum() == 0, "[!] ERROR - NodeEdgeCoupling: Found NaN values in output. Details:\n" + \
			   "\n".join(["-> %s: %s" % (name, str(torch.isnan(tensor).sum().item())) for name, tensor in \
			   			  [("z_nodes_out", z_nodes_out), ("z_edges_out", z_edges_out), ("ldj", ldj),
			   			   ("z_nodes", z_nodes), ("z_edges", z_edges), ("nn_nodes_out", nn_nodes_out),
			   			   ("nn_edges_out", nn_edges_out), ("nodes_ldj", nodes_ldj), ("edges_ldj", edges_ldj)]]) + "\n" + \
			   "\n".join(["=> %s: %s" % (name, str(tensor.data.detach().cpu().numpy() if tensor is not None else tensor)) for name, tensor in \
			   			  [("scaling nodes", self.scaling_factor_nodes), ("scaling edges", self.scaling_factor_edges),
			   			   ("mixture scaling nodes", self.mixture_scaling_factor_nodes), 
			   			   ("mixture scaling edges", self.mixture_scaling_factor_edges),
			   			   ("mixture add params nodes", self.mixture_add_params_nodes),
			   			   ("mixture add params edges", self.mixture_add_params_edges)]])

		detail_out = {"ldj": ldj}
		if nodes_reg_ldj is not None:
			detail_out["regularizer_nodes_ldj"] = nodes_reg_ldj
		if edges_reg_ldj is not None:
			detail_out["regularizer_edges_ldj"] = edges_reg_ldj

		return z_nodes_out, z_edges_out, ldj, detail_out


	def _run_mixture_layer(self, orig_z, nn_out, mask, num_mixtures, scaling_factor, 
								 mixture_scaling_factor, reverse, channel_padding_mask, **kwargs):
		mixture_params = MixtureCDFCoupling.get_mixt_params(nn_out=nn_out, 
															mask=mask, 
															num_mixtures=num_mixtures,
															scaling_factor=scaling_factor, 
															mixture_scaling_factor=mixture_scaling_factor)
		t, log_s, log_pi, mixt_t, mixt_log_s = mixture_params
		z_out, ldj, reg_ldj = MixtureCDFCoupling.run_with_params(orig_z=orig_z.double(), 
																  t=t, 
																  log_s=log_s, 
																  log_pi=log_pi, 
																  mixt_t=mixt_t,
																  mixt_log_s=mixt_log_s,
																  reverse=reverse,
																  is_training=self.training,
																  reg_max=self.regularizer_max,
																  reg_factor=self.regularizer_factor,
																  mask=mask,
																  channel_padding_mask=channel_padding_mask,
																  return_reg_ldj=True)
		z_out = z_out.float()
		ldj = ldj.float()
		if reg_ldj is not None:
			reg_ldj = reg_ldj.float().sum(dim=[i for i in range(1,len(reg_ldj.shape))])
		return z_out, ldj, reg_ldj


	def info(self):
		mask_ratio_nodes = self.mask_nodes.sum().item()/np.prod(self.mask_nodes.shape)
		mask_ratio_edges = self.mask_edges.sum().item()/np.prod(self.mask_edges.shape)
		return "Node+Edge Mixture Coupling Layer - Nodes: c_in=%i, num_mixtures=%2i, mask_ratio=%3.2f\n" % (self.c_in_nodes, self.num_mixtures_nodes, mask_ratio_nodes) + \
			   "                                   Edges: c_in=%i, num_mixtures=%2i, mask_ratio=%3.2f" % (self.c_in_edges, self.num_mixtures_edges, mask_ratio_edges)



class NodeEdgeFlowWrapper(FlowLayer):

	def __init__(self, node_flow, edge_flow):
		super().__init__()
		self.node_flow = node_flow
		self.edge_flow = edge_flow

	def forward(self, z_nodes, z_edges, ldj=None, reverse=False, length=None, channel_padding_mask=None, mask_valid=None, **kwargs):
		z_nodes, ldj = self.node_flow(z_nodes, ldj=ldj, reverse=reverse, length=length, channel_padding_mask=channel_padding_mask, **kwargs)
		
		edge_length = mask_valid.sum(dim=1)
		if len(mask_valid.shape) == 2:
			mask_valid = mask_valid.unsqueeze(dim=-1)
		z_edges, ldj = self.edge_flow(z_edges, ldj=ldj, reverse=reverse, length=edge_length, channel_padding_mask=mask_valid, **kwargs)
		return z_nodes, z_edges, ldj

	def need_data_init(self):
		return self.node_flow.need_data_init() or self.edge_flow.need_data_init()

	def data_init_forward(self, z_nodes, z_edges, channel_padding_mask=None, mask_valid=None, **kwargs):
		if self.node_flow.need_data_init():
			self.node_flow.data_init_forward(z_nodes, channel_padding_mask=channel_padding_mask)
		if self.edge_flow.need_data_init():
			if len(mask_valid.shape) == 2:
				mask_valid = mask_valid.unsqueeze(dim=-1)
			self.edge_flow.data_init_forward(z_edges, channel_padding_mask=mask_valid)

	def info(self):
		return "FlowWrapper - Node layer: %s\n" % self.node_flow.info() + \
			   "              Edge layer: %s" % self.edge_flow.info()