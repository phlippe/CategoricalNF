import torch
import torch.nn as nn 
import sys
sys.path.append("../../")

from general.train import TrainTemplate, get_default_train_arguments, start_training
from general.mutils import get_param_val, general_args_to_params
from general.parameter_scheduler import add_scheduler_parameters, scheduler_args_to_params
from layers.categorical_encoding.mutils import add_encoding_parameters, encoding_args_to_params
from layers.flows.distributions import add_prior_distribution_parameters, prior_distribution_args_to_params

from experiments.molecule_generation.task import TaskMoleculeGeneration
from experiments.molecule_generation.graphCNF import GraphCNF



class TrainMoleculeGeneration(TrainTemplate):

	
	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False, **kwargs):
		super().__init__(model_params, optimizer_params, batch_size, checkpoint_path, debug=debug, name_prefix="MoleculeGeneration_", **kwargs)


	def _create_model(self, model_params):
		dataset_name = get_param_val(self.model_params, "dataset", default_val="zinc250k")
		dataset_class = TaskMoleculeGeneration.get_dataset_class(dataset_name)
		model = GraphCNF(model_params, dataset_class)
		return model


	def _create_task(self, model_params, debug=False):
		task = TaskMoleculeGeneration(self.model, model_params, debug=debug, batch_size=self.batch_size)
		return task


def args_to_params(args):
	model_params, optimizer_params = general_args_to_params(args, model_params=dict())
	model_params["prior_distribution"] = prior_distribution_args_to_params(args)
	model_params["categ_encoding_nodes"] = encoding_args_to_params(args, postfix="_nodes")
	model_params["categ_encoding_edges"] = encoding_args_to_params(args, postfix="_edges")
	sched_params = scheduler_args_to_params(args, ["beta"])
	model_params.update(sched_params)
	dataset_params = {
		"dataset": args.dataset,
		"encoding_virtual_num_flows": args.encoding_virtual_num_flows
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
	add_encoding_parameters(parser, postfix="_nodes")
	add_encoding_parameters(parser, postfix="_edges")

	# Dataset parameters
	parser.add_argument("--dataset", help="Dataset to train on. Options: zinc250k, moses", type=str, default="zinc250k")
	
	# Additional encoding parameters
	parser.add_argument("--encoding_virtual_num_flows", help="Number of flows to use in the encoding of virtual edges.", type=int, default=0)
	
	# Coupling layer parameters
	parser.add_argument("--coupling_hidden_size_nodes", help="Hidden size of the coupling layers.", type=int, default=384)
	parser.add_argument("--coupling_hidden_size_edges", help="Hidden size of the coupling layers.", type=int, default=192)
	parser.add_argument("--coupling_hidden_layers", help="Number of hidden layers in the coupling layers. If specified as e.g. \"3,4,4\", first number refers to first stage etc.", type=str, default="3,4,4")
	parser.add_argument("--coupling_num_flows", help="Number of coupling layers to use for each step. Specify it in the format \"4,6,6\" where first number is for the first step etc.", type=str, default="4,6,6")
	parser.add_argument("--coupling_mask_ratio", help="Ratio of inputs masked out (i.e. for how many transformations are used) in the coupling layers.", type=float, default=0.5)
	parser.add_argument("--coupling_num_mixtures_nodes", help="Number of mixtures used in the coupling layers.", type=int, default=16)
	parser.add_argument("--coupling_num_mixtures_edges", help="Number of mixtures used in the coupling layers.", type=int, default=8)
	parser.add_argument("--coupling_dropout", help="Dropout to apply in the hidden layers.", type=float, default=0.0)
	
	# Parameter for schedulers
	add_scheduler_parameters(parser, ["beta"], 
							 {"beta": ("exponential", 1.0, 2.0, 5000, 2, 0)})

	# Parse given parameters and start training
	args = parser.parse_args()
	start_training(args, args_to_params, TrainMoleculeGeneration)