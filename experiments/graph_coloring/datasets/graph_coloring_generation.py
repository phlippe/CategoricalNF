import random
import numpy as np
import networkx as nx
import sys, os, json, argparse, itertools
import grinpy as gp
import time
from glob import glob
from multiprocessing import Pool
from ortools.sat.python import cp_model

"""
This code is based on https://github.com/machine-reasoning-ufrgs/GNN-GCP
"""


def solve_csp(M, n_colors, nmin=25):
  model = cp_model.CpModel()
  N = len(M)
  variables = []
  
  variables = [ model.NewIntVar(0, n_colors-1, '{i}'.format(i=i)) for i in range(N) ]
  
  for i in range(N):
    for j in range(i+1,N):
      if M[i][j] == 1:
        model.Add( variables[i] != variables [j] )
        
  solver = cp_model.CpSolver()
  solver.parameters.max_time_in_seconds = int( ((10.0 / nmin) * N) )
  status = solver.Solve(model)
  
  if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL :
    solution = dict()
    for k in range(N):
      solution[k] = solver.Value(variables[k])
    return solution
  elif status == cp_model.INFEASIBLE:
    return None
  else:
    raise Exception("CSP is unsure about the problem")


def is_cn(Ma, cn_i):
  if solve_csp(Ma, cn_i-1) == None:
    return True
  else:
    return False


def multiprocessing_dataset_generation(nmin, nmax, ncolors, path, samples, seed, num_workers=8):
	if not os.path.exists(path):
		os.makedirs(path)
	# For increasing sampling speed, we create multiple workers/processes in parallel
	samples_per_worker = int(samples//num_workers)
	p = Pool()
	args_list = [(nmin, nmax, ncolors, path, samples_per_worker, samples_per_worker*i, seed+i) for i in range(num_workers)]
	p.map(_create_simple_dataset_tuple, args_list)


def _create_simple_dataset_tuple(args):
	nmin, nmax, ncolors, path, samples, start_idx, seed = args
	create_simple_dataset(nmin, nmax, ncolors, path, samples, start_idx, seed)


def create_simple_dataset(nmin, nmax, ncolors, path, samples, start_idx=0, seed=123):
	if not os.path.exists(path):
		os.makedirs(path)
		
	export_pack = 500
	all_solutions = {"N": np.zeros((export_pack,), dtype=np.uint8), 
					 "adjacency": -np.ones((export_pack, nmax, nmax), dtype=np.int8), 
					 "nodes": -np.ones((export_pack, nmax), dtype=np.int8),
					 "graph_idx": -np.ones((export_pack,), dtype=np.int32),
					 "idx": 0}

	def export_solution(Ma, init_sol, z, graph_idx=-1):
		N, Ma, sol = write_solution(Ma=Ma, init_sol=init_sol, save_path=None)
		sol_idx = all_solutions["idx"]
		all_solutions["N"][sol_idx] = N
		all_solutions["adjacency"][sol_idx,:N,:N] = Ma.astype(np.uint8)
		all_solutions["nodes"][sol_idx,:N] = sol
		all_solutions["graph_idx"][sol_idx] = graph_idx

		all_solutions["idx"] += 1
		if all_solutions["idx"] >= export_pack:
			all_solutions.pop("idx")
			np.savez_compressed(os.path.join(path, "samples_%s_%s.npz" % (str(z-export_pack+2).zfill(7), str(z+1).zfill(7))), 
								**all_solutions)

			all_solutions["N"].fill(0)
			all_solutions["adjacency"].fill(-1)
			all_solutions["nodes"].fill(-1)
			all_solutions["graph_idx"].fill(-1)
			all_solutions["idx"] = 0

	# Adjacency density ratio to sample from. 
	edge_prob_constraints = {3: (0.1, 0.3), 4: (0.15, 0.3)}

	np.random.seed(seed)
	random.seed(seed)
	z = start_idx
	N = np.random.randint(nmin, nmax+1)
	while z in range(start_idx,samples+start_idx):
		N = np.random.randint(nmin, nmax+1)
		save_path = os.path.join(path, "sample_%s.npz" % (str(z).zfill(6)))
		found_sol = False
		
		Cn = ncolors
		lim_inf, lim_sup = edge_prob_constraints[Cn][0], edge_prob_constraints[Cn][1]
		lim_sup = min(lim_sup, nmax/N*(lim_inf+lim_sup)/2.0)

		p_connected = random.uniform(lim_inf, lim_sup)
		Ma = gen_matrix(N, p_connected)

		init_sol = solve_csp(Ma, Cn)
		if init_sol is not None and is_cn(Ma,Cn):
			export_solution(Ma, init_sol, z)
			found_sol = True

		if found_sol:
			z += 1
			if z % 100 == 0:
				print("Completed %i (%4.2f%%) in [%i,%i] samples..." % (z-start_idx, (z-start_idx)*100.0/samples, start_idx, start_idx+samples))


def write_solution(Ma, init_sol, save_path=None):
	N = Ma.shape[0]
	sol = np.zeros(N, dtype=np.uint8)
	for i in range(N):
		sol[i] = int(init_sol[i])
	if save_path is not None:
		np.savez_compressed(save_path, adjacency=Ma, nodes=sol)
	else:
		return (N, Ma, sol)


def combine_solution_files(save_path):
	print("Combining solution files...")
	sample_files = sorted(glob(os.path.join(save_path, "sample*.npz")))
	nodes, adjacency = None, None
	for filename in sample_files:
		data_arr = np.load(filename)
		if nodes is None and adjacency is None:
			nodes, adjacency = data_arr["nodes"], data_arr["adjacency"]
		else:
			nodes = np.concatenate([nodes, data_arr["nodes"]], axis=0)
			adjacency = np.concatenate([adjacency, data_arr["adjacency"]], axis=0)
	np.savez_compressed(os.path.join(save_path, "samples_combined.npz"), nodes=nodes, adjacency=adjacency)


def gen_matrix(N, prob):
	Ma = np.zeros((N,N))
	Ma = np.random.choice([0,1], size=(N, N), p=[1-prob,prob])
	i_lower = np.tril_indices(N, -1)
	Ma[i_lower] = Ma.T[i_lower]  # make the matrix symmetric
	np.fill_diagonal(Ma, 0)

	# Ensuring that every node has at least 1 connection
	while np.min(Ma.sum(axis=0)) == 0:
		idx = np.argmin(Ma.sum(axis=0))
		Ma[idx,:] = np.random.choice([0,1], size=(N,), p=[1-prob,prob])
		Ma[:,idx] = Ma[idx,:]
		Ma[idx,idx] = 0

	# Test that the whole graph is connected
	connect = np.zeros((N,))
	connect[0] = 1
	Ma_diag = np.eye(N) + Ma
	while (1 - connect).sum() > 0:
		new_connect = ((connect[None,:] * Ma_diag).sum(axis=1) > 0).astype(connect.dtype)
		if np.any(new_connect != connect):
			connect = new_connect
		else:
			num_choices = 3
			start_nodes = np.random.choice(np.where(connect>0)[0], size=(num_choices,))
			end_nodes = np.random.choice(np.where(connect==0)[0], size=(num_choices,))
			Ma[start_nodes, end_nodes] = 1
			Ma[end_nodes, start_nodes] = 1
			Ma_diag = np.eye(N) + Ma

	return Ma


if __name__ == '__main__':
	# Define argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Path to which the files should be saved.', type=str, required=True)
	parser.add_argument('--samples', help='Number of samples to generate', type=int, default=2e5)
	parser.add_argument('--nmin', default=25, help='Minimum number of nodes in a graph', type=int)
	parser.add_argument('--nmax', default=50, help='Maximum number of nodes in a graph', type=int)
	parser.add_argument('--ncolor', default=3, help='Number of colors to use for the graph coloring', type=int)
	parser.add_argument('--train', help='If train is selected, we use a different seed', action='store_true')

	# Parse arguments from command line
	args = parser.parse_args()
	seed = 1327 if args.train else 3712
	random.seed(seed)
	np.random.seed(seed)

	# Start the generation process
	print('Creating {} instances'.format(args.samples))
	multiprocessing_dataset_generation(
			args.nmin, args.nmax,
			ncolors=args.ncolor,
			samples=args.samples,
			path=args.path,
			seed=seed
		)
	combine_solution_files(args.path)