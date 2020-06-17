import torch
import torch.nn as nn 
import numpy as np
import argparse
import os
import sys
sys.path.append("../../")

from general.mutils import load_args
from experiments.molecule_generation.train import TrainMoleculeGeneration, args_to_params
from experiments.molecule_generation.datasets.mutils import analyse_generations


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_path", help="Folder(name) where the checkpoints and parameter config are saved.", type=str, required=True)
	
	args = parser.parse_args()
	checkpoint_path = args.checkpoint_path
	args = load_args(checkpoint_path)
	args.checkpoint_path = checkpoint_path

	model_params, optimizer_params = args_to_params(args)
	trainModule = TrainMoleculeGeneration(model_params=model_params,
										  optimizer_params=optimizer_params, 
										  batch_size=args.batch_size,
										  checkpoint_path=args.checkpoint_path, 
										  debug=False,
										  multi_gpu=False)

	sample_file = os.path.join(checkpoint_path, "molecule_samples_best.npz")
	if os.path.isfile(sample_file):
		print("\nAnalysing the saved samples at %s...\n" % sample_file)
		samples = np.load(sample_file)
		nodes, adjacency, length = samples["z"], samples["adjacency"], samples["length"]
		analyse_generations(trainModule.task.dataset_class, nodes, adjacency, length)

	print("Evaluating the model on the test set and generating new molecules...")
	trainModule.evaluate_model()