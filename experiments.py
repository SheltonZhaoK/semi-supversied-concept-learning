import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,4,6"
import numpy as np
import pdb, argparse
import sys, random
import ray, copy, torch
from ray import tune
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.configs.mi_estimator_config import MI_ESTIMATOR

def run_experiments(dataset, args, mi_args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if dataset == "CUB":
        from CUB.new_train import (train_X_to_C_to_y)
    if dataset == "HAM10K":
         from CUB.train_ham10k import (train_X_to_C_to_y)
    
    train_X_to_C_to_y(args, mi_args)

def parse_arguments():
    #First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, 'You need to specify dataset and experiment'
    assert sys.argv[1].upper() in ['HAM10K', 'CUB'], 'Please specify HAM10K or CUB dataset'
    assert sys.argv[2] in ['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                           'Standard', 'StandardWithAuxC', 'Multitask', 'Joint', 'Probe',
                           'TTI', 'Robustness', 'HyperparameterSearch'], \
        'Please specify valid experiment. Current: %s' % sys.argv[2]
    dataset = sys.argv[1].upper()
    experiment = sys.argv[2].upper()

    # # # Handle accordingly to dataset
    if dataset == 'HAM10K':
        from CUB.train_ham10k import parse_arguments
    elif dataset == 'CUB':
        from CUB.new_train import parse_arguments

    args = parse_arguments(experiment=experiment)
    return dataset, args

def rayTune_run_experiment(config, checkpoint_dir=None):
    args = config["args"]
    dataset = config["dataset"]
    mi_args = argparse.Namespace(**config["mi_args"])
    run_experiments(dataset, args, mi_args)

def create_args_list(args, search_space):
    arg_list = [copy.deepcopy(args) for _ in range(len(search_space[0]) * len(search_space[1]))]

    index = 0
    for i in range(len(search_space[0])):
        for j in range(len(search_space[1])):
            arg_list[index].seed = search_space[0][i]
            arg_list[index].subset = search_space[1][j]
            arg_list[index].log_dir = os.path.join(arg_list[index].log_dir, str(search_space[1][j]))
            arg_list[index].n_attributes = search_space[1][j]
            index += 1
    return arg_list

if __name__ == '__main__':
    dataset, args = parse_arguments()

    if args.ray:
        if args.dataset == "cub":
            # arg_list = create_args_list(args, [[1], [20,40,60,80,100,112]])
            arg_list = create_args_list(args, [[1], [80,100,112]])
        if args.dataset == "ham10k":
            arg_list = create_args_list(args, [[1], [4]])
            
        config = {
            "args": tune.grid_search(arg_list),
            "dataset": dataset,
            "mi_args": MI_ESTIMATOR().config
        }
        
        tuner = tune.run(
            tune.with_resources(tune.with_parameters(rayTune_run_experiment), {"cpu": 1, "gpu": 1}),
            config = config
        )
    elif args.ray_tune:
        config = {
            "args": args,
            "dataset": dataset,
            "mi_args": MI_ESTIMATOR().config
        }

        tuner = tune.run(
            tune.with_resources(tune.with_parameters(rayTune_run_experiment), {"cpu": 1, "gpu": 1}),
            config = config
        )
    else:
        mi_args = argparse.Namespace(**MI_ESTIMATOR().config)
        run_experiments(dataset, args, mi_args)
