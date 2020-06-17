import torch
import torch.nn as nn 
import sys
sys.path.append("../../")

from general.train import TrainTemplate, get_default_train_arguments, start_training
from general.mutils import get_param_val, general_args_to_params
from general.parameter_scheduler import add_scheduler_parameters, scheduler_args_to_params
from layers.categorical_encoding.mutils import add_encoding_parameters, encoding_args_to_params
from layers.flows.distributions import add_prior_distribution_parameters, prior_distribution_args_to_params

from experiments.graph_coloring.task import TaskGraphColoring
from experiments.graph_coloring.graph_node_flow import GraphNodeFlow
from experiments.graph_coloring.graph_node_rnn import GraphNodeRNN
from experiments.graph_coloring.graph_node_vae import GraphNodeVAE



class TrainGraphColoring(TrainTemplate):

	
	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False, **kwargs):
		super().__init__(model_params, optimizer_params, batch_size, checkpoint_path, debug=debug, name_prefix="GraphColoring_", **kwargs)


	def _create_model(self, model_params):
		dataset_name = get_param_val(self.model_params, "dataset", default_val="tiny_3")
		dataset_class = TaskGraphColoring.get_dataset_class(dataset_name)

		use_rnn = get_param_val(model_params, "use_rnn", default_val=False)
		use_vae = get_param_val(model_params, "use_vae", default_val=False)
		if use_rnn:
			model = GraphNodeRNN(model_params, dataset_class)
		elif use_vae:
			model = GraphNodeVAE(model_params, dataset_class)
		else:
			model = GraphNodeFlow(model_params, dataset_class)

		return model


	def _create_task(self, model_params, debug=False):
		task = TaskGraphColoring(self.model, model_params, debug=debug, batch_size=self.batch_size)
		return task


def args_to_params(args):
	model_params, optimizer_params = general_args_to_params(args, model_params=dict())
	model_params["prior_distribution"] = prior_distribution_args_to_params(args)
	model_params["categ_encoding"] = encoding_args_to_params(args)
	sched_params = scheduler_args_to_params(args, ["beta", "gamma"])
	model_params.update(sched_params)
	dataset_params = {
		"use_rnn": args.use_rnn,
		"use_vae": args.use_vae,
		"dataset": args.dataset,
		"rnn_graph_ordering": args.rnn_graph_ordering
	}
	coupling_params = {p_name: getattr(args, p_name) for p_name in vars(args) if p_name.startswith("coupling_")}
	model_params.update(coupling_params)
	model_params.update(dataset_params)
	return model_params, optimizer_params


if __name__ == "__main__":
	parser = get_default_train_arguments()
	
	# Add parameters for prior distribution
	add_prior_distribution_parameters(parser)

	# Add parameters for categorical encoding
	add_encoding_parameters(parser)

	# Selection of the model
	parser.add_argument("--use_rnn", help="If selected, a graph RNN model is used instead of a normalizing flow", action="store_true")
	parser.add_argument("--use_vae", help="If selected, a graph VAE model is used instead of a normalizing flow", action="store_true")

	# Dataset parameters
	parser.add_argument("--dataset", help="Dataset to train on, as \"size_numcolors\". Options: tiny_3, large_3", type=str, default="tiny_3")
	parser.add_argument("--rnn_graph_ordering", help="Ordering to use for a RNN approach. Options: rand, smallest_first, largest_first.", default="rand")

	# Coupling layer parameters
	parser.add_argument("--coupling_hidden_size", help="Hidden size of the coupling layers.", type=int, default=384)
	parser.add_argument("--coupling_hidden_layers", help="Number of hidden layers in the coupling layers.", type=int, default=4)
	parser.add_argument("--coupling_num_flows", help="Number of coupling layers to use.", type=int, default=8)
	parser.add_argument("--coupling_mask_ratio", help="Ratio of inputs masked out (i.e. for how many transformations are used) in the coupling layers.", type=float, default=0.5)
	parser.add_argument("--coupling_num_mixtures", help="Number of mixtures used in the coupling layers.", type=int, default=16)
	parser.add_argument("--coupling_dropout", help="Dropout to apply in the hidden layers.", type=float, default=0.0)
	
	# Parameter for schedulers
	add_scheduler_parameters(parser, ["beta", "gamma"], 
							 {"beta": ("exponential", 1.0, 2.0, 5000, 2, 0),
							  "gamma": ("sigmoid", 1.0, 0.1, 15000, 0.0003, 0)})

	# Parse given parameters and start training
	args = parser.parse_args()
	start_training(args, args_to_params, TrainGraphColoring)