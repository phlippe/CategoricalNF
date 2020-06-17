import torch
import torch.nn as nn 
import sys
sys.path.append("../../")

from general.train import TrainTemplate, get_default_train_arguments, start_training
from general.mutils import get_param_val, general_args_to_params
from general.parameter_scheduler import add_scheduler_parameters, scheduler_args_to_params
from layers.categorical_encoding.mutils import add_encoding_parameters, encoding_args_to_params
from layers.flows.distributions import add_prior_distribution_parameters, prior_distribution_args_to_params

from experiments.set_modeling.task import TaskSetModeling
from experiments.set_modeling.flow_model import FlowSetModeling
from experiments.set_modeling.discrete_flow import DiscreteFlowSetModeling


class TrainSetModeling(TrainTemplate):

	
	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False, **kwargs):
		super().__init__(model_params, optimizer_params, batch_size, checkpoint_path, debug=debug, name_prefix="SetModeling_", **kwargs)


	def _create_model(self, model_params):
		dataset_name = get_param_val(self.model_params, "dataset", default_val="shuffling")
		dataset_class = TaskSetModeling.get_dataset_class(dataset_name)
		use_discrete = get_param_val(self.model_params, "use_discrete", default_val=False)
		if use_discrete:
			model = DiscreteFlowSetModeling(model_params=model_params, dataset_class=dataset_class)
		else:
			model = FlowSetModeling(model_params=model_params, dataset_class=dataset_class)
		return model


	def _create_task(self, model_params, debug=False):
		task = TaskSetModeling(self.model, model_params, debug=debug, batch_size=self.batch_size)
		return task


def args_to_params(args):
	model_params, optimizer_params = general_args_to_params(args, model_params=dict())
	model_params["prior_distribution"] = prior_distribution_args_to_params(args)
	model_params["categ_encoding"] = encoding_args_to_params(args)
	sched_params = scheduler_args_to_params(args, ["beta"])
	model_params.update(sched_params)
	dataset_params = {
		"set_size": args.set_size,
		"dataset": args.dataset,
		"use_discrete": args.use_discrete
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

	# Dataset parameters
	parser.add_argument("--set_size", help="Number of elements in the sets.", type=int, default=16)
	parser.add_argument("--dataset", help="Name of the dataset to train on. Options: shuffling, summation", type=str, default="shuffling")
	parser.add_argument("--use_discrete", help="If selected, a discrete normalizing flow is trained.", action="store_true")
	
	# Coupling layer parameters
	parser.add_argument("--coupling_hidden_size", help="Hidden size of the coupling layers.", type=int, default=256)
	parser.add_argument("--coupling_hidden_layers", help="Number of hidden layers in the coupling layers.", type=int, default=2)
	parser.add_argument("--coupling_num_flows", help="Number of coupling layers to use.", type=int, default=8)
	parser.add_argument("--coupling_mask_ratio", help="Ratio of inputs masked out (i.e. for how many mu/scale transformations are used) in the coupling layers.", type=float, default=0.5)
	parser.add_argument("--coupling_num_mixtures", help="Number of mixtures used in the coupling layers.", type=int, default=8)
	
	# Parameter for schedulers
	add_scheduler_parameters(parser, ["beta"], 
							 {"beta": ("exponential", 1.0, 2.0, 5000, 2, 0)})

	# Parse given parameters and start training
	args = parser.parse_args()
	start_training(args, args_to_params, TrainSetModeling)