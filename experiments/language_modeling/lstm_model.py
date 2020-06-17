import torch
import torch.nn as nn 
import torch.nn.utils.weight_norm as weight_norm
import numpy as np 
import sys
sys.path.append("../../")
from general.mutils import get_param_val, _create_length_mask
from layers.networks.autoregressive_layers import TimeConcat


class LSTMModel(nn.Module):


	def __init__(self, num_classes, hidden_size=64, num_layers=2, embedding_dim=32, dp_rate=0.0, input_dp_rate=0.0,
					   max_seq_len=-1, vocab=None, model_params=None):
		super().__init__()
		if model_params is not None:
			hidden_size = get_param_val(model_params, "coupling_hidden_size", hidden_size)
			embedding_dim = hidden_size//4
			num_layers = get_param_val(model_params, "coupling_hidden_layers", num_layers)
			dp_rate = get_param_val(model_params, "coupling_dropout", dp_rate)
			input_dp_rate = get_param_val(model_params, "coupling_input_dropout", input_dp_rate)
			max_seq_len = get_param_val(model_params, "max_seq_len", max_seq_len)

		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.embed_dim = 1 # Not equal to embedding_dim, is needed for making sampling equal to flows

		if vocab is not None and vocab.vectors is not None:
			embedding_dim = vocab.vectors.shape[1]
			self.embeddings = nn.Embedding(num_embeddings=len(vocab),
										   embedding_dim=embedding_dim)
			self.embeddings.weight.data.copy_(vocab.vectors)
			self.vocab_size = len(vocab)
		else:
			self.embeddings = nn.Embedding(num_embeddings=num_classes,
										   embedding_dim=embedding_dim)
			self.vocab_size = num_classes

		if time_dp_rate < 1.0:
			time_embed_dim = embedding_dim//4
			time_embed = nn.Linear(2*max_seq_len, time_embed_dim)
			self.max_seq_len = max_seq_len
			self.time_concat = TimeConcat(time_embed=time_embed, 
										  input_dp_rate=input_dp_rate)
		else:
			self.time_concat = None
			time_embed_dim = 0
		self.lstm = nn.LSTM(input_size=embedding_dim+time_embed_dim, 
							hidden_size=hidden_size, 
							num_layers=num_layers, 
							batch_first=True,
							bidirectional=False)

		self.init_state = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))

		self.output_layer = nn.Sequential(
				nn.Linear(hidden_size, hidden_size//2),
				nn.GELU(),
				nn.Dropout(dp_rate),
				nn.Linear(hidden_size//2, num_classes),
				nn.LogSoftmax(dim=-1)
			)


	def forward(self, x, reverse=False, **kwargs):
		if not reverse:
			return self._determine_logp(x, **kwargs)
		else:
			return self._sample(x.size(0), x.size(1))


	def _determine_logp(self, x, length=None, channel_padding_mask=None, **kwargs):
		if length is not None and channel_padding_mask is None:
			channel_padding_mask = _create_length_mask(length, max_len=x.shape[1])
		x_embed = self.embeddings(x)
		if self.time_concat is not None:
			x_embed = self.time_concat(x=x_embed, length=length)
		init_state = self.init_state.expand(-1, x.size(0), -1).contiguous()
		h_0 = torch.tanh(init_state)
		c_0 = init_state

		x_prior = self.output_layer(h_0[0,:,:]) # Use initial state to predict prior
		x_hidden, (h_n, c_n) = self.lstm(x_embed, (h_0, c_0))
		x_pred = self.output_layer(x_hidden)
		logp = torch.cat([x_prior[:,None,:].gather(index=x[...,:1,None], dim=2),
						  x_pred[:,:-1,:].gather(index=x[...,1:,None], dim=2)], dim=1)

		logp = logp.sum(dim=2) # Shape: [Batch,Seq_len]
		if channel_padding_mask is not None:
			channel_padding_mask = channel_padding_mask.squeeze(dim=-1)
			logp = (logp * channel_padding_mask) / channel_padding_mask.sum(dim=[1], keepdims=True)

		ldj = logp.sum(dim=1)

		with torch.no_grad():
			detailed_log = {
				"avg_token_prob": -logp.exp().mean(dim=1),
				"avg_token_bpd": logp.mean(dim=1) * np.log2(np.exp(1))
			}

		return ldj, detailed_log


	def _sample(self, batch_size, seqlen):
		init_state = self.init_state.expand(-1, batch_size, -1).contiguous()
		h_0 = torch.tanh(init_state)
		c_0 = init_state

		preds = []
		x_prior = self.output_layer(h_0[0,:,:]).exp()
		preds.append(torch.multinomial(input=x_prior, num_samples=1).squeeze(dim=-1))
		h_n, c_n = h_0, c_0
		for _ in range(seqlen-1):
			x_embed = self.embeddings(preds[-1])
			x_hidden, (h_n, c_n) = self.lstm(x_embed.unsqueeze(dim=1), (h_n, c_n))
			x_probs = self.output_layer(x_hidden.squeeze(dim=1)).exp()
			preds.append(torch.multinomial(input=x_probs, num_samples=1).squeeze(dim=-1))
		return torch.stack(preds, dim=1), None


	def need_data_init(self):
		return False


	def info(self):
		return "LSTM with hidden size %i and %i layers" % (self.hidden_size, self.num_layers)
