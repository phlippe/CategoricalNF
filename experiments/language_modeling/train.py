import torch
import torch.nn as nn 
import sys
sys.path.append("../../")

from general.train import TrainTemplate, get_default_train_arguments, start_training
from general.mutils import get_param_val, general_args_to_params
from general.parameter_scheduler import add_scheduler_parameters, scheduler_args_to_params
from layers.categorical_encoding.mutils import add_encoding_parameters, encoding_args_to_params
from layers.flows.distributions import add_prior_distribution_parameters, prior_distribution_args_to_params

from experiments.language_modeling.task import TaskLanguageModeling
from experiments.language_modeling.lstm_model import LSTMModel
from experiments.language_modeling.flow_model import FlowLanguageModeling


class TrainLanguageModeling(TrainTemplate):

	
	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False, **kwargs):
		super().__init__(model_params, optimizer_params, batch_size, checkpoint_path, debug=debug, name_prefix="LanguageModeling_", **kwargs)


	def _create_model(self, model_params):
		dataset_name = get_param_val(self.model_params, "dataset", default_val="penntreebank")
		dataset_class = TaskLanguageModeling.get_dataset_class(dataset_name)
		vocab_dict = dataset_class.get_vocabulary()
		vocab_torchtext = dataset_class.get_torchtext_vocab()

		use_rnn = get_param_val(self.model_params, "use_rnn", default_val=False)
		if use_rnn:
			model = LSTMModel(num_classes=len(vocab_dict), vocab=vocab_torchtext, model_params=model_params)
		else:
			model = FlowLanguageModeling(model_params=model_params, vocab_size=len(vocab_dict), vocab=vocab_torchtext, dataset_class=dataset_class)
		return model


	def _create_task(self, model_params, debug=False):
		task = TaskLanguageModeling(self.model, model_params, debug=debug, batch_size=self.batch_size)
		return task



def args_to_params(args):
	model_params, optimizer_params = general_args_to_params(args, model_params=dict())
	model_params["prior_distribution"] = prior_distribution_args_to_params(args)
	model_params["categ_encoding"] = encoding_args_to_params(args)
	sched_params = scheduler_args_to_params(args, ["beta"])
	model_params.update(sched_params)
	dataset_params = {
		"max_seq_len": args.max_seq_len,
		"dataset": args.dataset,
		"use_rnn": args.use_rnn
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
	parser.add_argument("--max_seq_len", help="Maximum sequence length of training sentences.", type=int, default=256)
	parser.add_argument("--dataset", help="Name of the dataset to train on. Options: penntreebank, text8, wikitext", type=str, default="penntreebank")
	parser.add_argument("--use_rnn", help="If selected, a RNN is used as model instead of a Categorical Normalizing Flow", action="store_true")

	# Coupling layer parameters
	parser.add_argument("--coupling_hidden_size", help="Hidden size of the coupling layers.", type=int, default=1024)
	parser.add_argument("--coupling_hidden_layers", help="Number of hidden layers in the coupling layers.", type=int, default=2)
	parser.add_argument("--coupling_num_flows", help="Number of coupling layers to use.", type=int, default=1)
	parser.add_argument("--coupling_num_mixtures", help="Number of mixtures used in the coupling layers.", type=int, default=64)
	parser.add_argument("--coupling_dropout", help="Dropout to use in the networks.", type=float, default=0.0)
	parser.add_argument("--coupling_input_dropout", help="Input dropout rate to use in the networks.", type=float, default=0.0)
	
	# Parameter for schedulers
	add_scheduler_parameters(parser, ["beta"], 
							 {"beta": ("exponential", 1.0, 2.0, 5000, 2, 0)})

	# Parse given parameters and start training
	args = parser.parse_args()
	start_training(args, args_to_params, TrainLanguageModeling)