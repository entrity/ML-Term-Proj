import os
import logging
import argparse

def is_path_ok(path):
	return path is not None and len(path) and os.path.exists(path)

def init_log(log_path):
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')
	logging.getLogger().addHandler(logging.StreamHandler())

def default_arg_parser():
	parser = argparse.ArgumentParser()
	paths = parser.add_argument_group('Paths')
	paths.add_argument('--train', default='data/train')
	paths.add_argument('--test', default='data/test')
	paths.add_argument('-l', '--log_path', '--l')
	paths.add_argument('-m', '--load_path', '--m')
	paths.add_argument('-s', '--save_path', '--s')
	periods = parser.add_argument_group('Periods')
	periods.add_argument('--print_every', type=int, default=100)
	periods.add_argument('--test_every', type=int, default=100)
	periods.add_argument('--save_every', type=int, default=1, help='Save a checkpoint after every *k* epochs. 0 will disable these periodic saves')
	hyperparams = parser.add_argument_group('Hyperparameters')
	hyperparams.add_argument('--lr', type=float, default=1e-2)
	hyperparams.add_argument('--train_bs', type=int, default=64)
	hyperparams.add_argument('--test_bs', type=int, default=64)
	hyperparams.add_argument('--ep', default=1000, type=int, help='Max epochs to train')
	others = parser.add_argument_group('Other')
	others.add_argument('-c', '--do_continue', action='store_true', help='Dictates whether to load optim dict, scheduler dict, epoch_i')
	return parser

def init(arg_parser_fn=None):
	# Prepare default argument parser
	parser = default_arg_parser()
	# Add task-specific arguments
	if arg_parser_fn is not None:
		arg_parser_fn(parser)
	# Parse args
	args = parser.parse_args()
	assert args.save_path is not None
	# Initialize logger
	init_log(args.log_path)
	logging.info(args)
	# Return
	return args
