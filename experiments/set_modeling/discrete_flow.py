import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append("../../")

from general.mutils import one_hot, get_param_val, create_transformer_mask, create_channel_mask
from layers.flows.flow_model import FlowModel 
from layers.flows.coupling_layer import CouplingLayer
from layers.flows.discrete_coupling import DiscreteCouplingLayer


class DiscreteFlowSetModeling(FlowModel):


	def __init__(self, model_params, dataset_class):
		super().__init__(layers=None, name="Set Modeling Discrete Flow")
		self.model_params = model_params
		self.dataset_class = dataset_class
		self.set_size = self.model_params["set_size"]
		self.vocab_size = self.dataset_class.get_vocab_size(self.set_size)

		self._create_layers()
		self.print_overview()


	def _create_layers(self):
		model_func = lambda c_out : CouplingTransformerNet(vocab_size=self.vocab_size,
														   c_out=c_out,
														   num_layers=self.model_params["coupling_hidden_layers"],
														   hidden_size=self.model_params["coupling_hidden_size"])
		coupling_mask = CouplingLayer.create_chess_mask()
		coupling_mask_func = lambda flow_index : coupling_mask if flow_index%2==0 else 1-coupling_mask
		num_flows = self.model_params["coupling_num_flows"]

		layers = []
		for flow_index in range(num_flows):
			layers += [
				DiscreteCouplingLayer(c_in=self.vocab_size,
									  mask=coupling_mask_func(flow_index),
									  model_func=model_func,
									  block_type="Transformer",
									  temp=0.1)
			]

		self.flow_layers = nn.ModuleList(layers)

		self.prior = nn.Parameter(0.2 * torch.randn(self.set_size, self.vocab_size), requires_grad=True)


	def forward(self, z, ldj=None, reverse=False, length=None, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0), dtype=torch.float32)
		if length is not None:
			kwargs["src_key_padding_mask"] = create_transformer_mask(length)
			kwargs["channel_padding_mask"] = create_channel_mask(length)

		if not reverse:
			z = one_hot(z, num_classes=self.vocab_size)
			for flow in self.flow_layers:
				z, ldj = flow(z, ldj, reverse=reverse, length=length, **kwargs)
			prior = F.log_softmax(self.prior, dim=-1)
			ldj = (z * prior[None,:z.size(1)] * kwargs["channel_padding_mask"]).sum(dim=[1,2])
		else:
			for flow in reversed(self.flow_layers):
				z, ldj = flow(z, ldj, reverse=reverse, length=length, **kwargs)
		return z, ldj



class CouplingTransformerNet(nn.Module):

	def __init__(self, vocab_size, c_out, num_layers, hidden_size):
		super().__init__()
		self.input_layer = nn.Embedding(vocab_size, hidden_size)
		self.transformer_layers = nn.ModuleList([
				nn.TransformerEncoderLayer(hidden_size, 
										   nhead=4, 
										   dim_feedforward=2*hidden_size, 
										   dropout=0.0, 
										   activation='gelu') for _ in range(num_layers)
			])
		self.output_layer = nn.Sequential(
				nn.LayerNorm(hidden_size),
				nn.Linear(hidden_size, hidden_size),
				nn.GELU(),
				nn.Linear(hidden_size, c_out)
			)

	def forward(self, x, src_key_padding_mask, **kwargs):
		x = x.transpose(0, 1) # Transformer layer expects [Sequence length, Batch size, Hidden size]
		x = self.input_layer(x.argmax(dim=-1))
		for transformer in self.transformer_layers:
			x = transformer(x, src_key_padding_mask=src_key_padding_mask)
		x = self.output_layer(x)
		x = x.transpose(0, 1)
		return x