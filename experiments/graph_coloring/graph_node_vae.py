import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append("../../")

# General imports
from general.mutils import get_param_val, one_hot, create_transformer_mask, create_channel_mask
from layers.flows.flow_model import FlowModel 
# Graph specific imports
from layers.networks.graph_layers import RGCNNet, RelationGraphAttention, RelationGraphConv


class GraphNodeVAE(FlowModel):

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
		self.latent_dim = 4
		self.embed_dim = self.latent_dim
		self.num_node_types = self.dataset_class.num_node_types()

		self.hidden_size = get_param_val(self.model_params, "coupling_hidden_size", default_val=512)
		self.hidden_layers = get_param_val(self.model_params, "coupling_hidden_layers", default_val=4)
		dropout = get_param_val(self.model_params, "coupling_dropout", default_val=0.0)

		self.embed_layer = nn.Embedding(self.num_node_types, self.hidden_size)
		self.graph_encoder = RGCNNet(c_in=self.hidden_size,
									 c_out=2*self.latent_dim,
									 num_edges=1,
									 num_layers=self.hidden_layers,
									 hidden_size=self.hidden_size,
									 dp_rate=dropout,
									 rgc_layer_fun=RelationGraphAttention)
		self.graph_decoder = RGCNNet(c_in=self.latent_dim,
									 c_out=self.num_node_types,
									 num_edges=1,
									 num_layers=self.hidden_layers,
									 hidden_size=self.hidden_size,
									 dp_rate=dropout,
									 rgc_layer_fun=RelationGraphAttention)


	###################
	## RNN Execution ##
	###################

	def forward(self, z, adjacency=None, ldj=None, reverse=False, get_ldj_per_layer=False, length=None, sample_temp=1.0, gamma=1.0, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0), dtype=torch.float32)
		if length is not None:
			kwargs["length"] = length
			kwargs["src_key_padding_mask"] = create_transformer_mask(length, max_len=z.size(1))
			kwargs["channel_padding_mask"] = create_channel_mask(length, max_len=z.size(1))

		z_nodes = None
		ldj_per_layer = []
		if not reverse:
			orig_nodes = z
			## Encoder
			z_embed = self.embed_layer(z)
			z_enc = self.graph_encoder(z_embed, adjacency, **kwargs)
			z_mu, z_log_std = z_enc.chunk(2, dim=-1)
			z_std = z_log_std.exp()
			z_latent = torch.randn_like(z_mu) * z_std + z_mu
			## Decoder
			z_dec = self.graph_decoder(z_latent, adjacency, **kwargs)
			z_rec = F.log_softmax(z_dec, dim=-1)
			## Loss calculation
			loss_mask = kwargs["channel_padding_mask"].squeeze(dim=-1)
			reconstruction_loss = F.nll_loss(z_rec.view(-1, self.num_node_types), z.view(-1), reduction='none').view(z.shape) * loss_mask
			reconstruction_loss = reconstruction_loss.sum(dim=-1) / length.float()
			KL_div = (- z_log_std + (z_std ** 2 - 1 + z_mu ** 2) / 2).sum(dim=-1) * loss_mask
			KL_div = KL_div.sum(dim=-1) / length.float()

			ldj = ldj - (reconstruction_loss + (gamma * KL_div + (1-gamma) * KL_div.detach()))
			ldj_per_layer.append({"KL_div": -KL_div, "reconstruction_loss": -reconstruction_loss})
			return ldj, ldj_per_layer
		else:
			z_latent = torch.randn_like(z)
			## Decoder
			z_dec = self.graph_decoder(z_latent, adjacency, **kwargs)
			## Sampling
			z_dec = z_dec / sample_temp
			out_pred = torch.softmax(z_dec, dim=-1)
			z_nodes = torch.multinomial(out_pred.view(-1,out_pred.size(-1)), num_samples=1, replacement=True).view(out_pred.shape[:-1])
			return z_nodes, None

	def test_reversibility(self, *args, **kwargs):
		return True

	def initialize_data_dependent(self, **kwargs):
		pass

	def need_data_init(self):
		return False

	def print_overview(self):
		# Retrieve layer descriptions for all flows
		s = "="*30 + "\n"
		s += "GraphNodeVAE network\n"
		s += "-"*30 + "\n"
		s += "-> Hidden size: %i\n" % self.hidden_size
		s += "-> Num layers: %i\n" % self.hidden_layers
		s += "="*30 
		print(s)


if __name__ == '__main__':
	pass
