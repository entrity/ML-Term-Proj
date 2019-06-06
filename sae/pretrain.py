import os, torch, logging
from shared import util
from shared import trainer
from . import net
import sae.dataset

class Trainer(trainer.Trainer):	
	def save_model(self, suffix=None):
		if suffix is not None:
			pre, post = os.path.splitext(self.save_path)
			save_path = pre + suffix + post
		else:
			save_path = self.save_path
		torch.save({
			'state_dict': self.net.state_dict(),
			'optim_dict': self.optim.state_dict(),
			'sched_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
			'best_test': self.best_test,
			'best_acc': self.best_acc,
			'iter_i': self.iter_i,
			'epoch_i': self.epoch_i,
			'batch_i': self.batch_i,
			'arch': self.net.arch,
			}, save_path)

def load_model(load_path):
	dump = torch.load(load_path)
	model = net.Net(arch=dump['arch'])
	model.load_state_dict( dump['state_dict'] )
	return model

def add_args(parser):
	parser.add_argument('--arch', default=[1296,500,500,2000,10], nargs='+', type=int, help='Dimensions for stacked autoencoders')
	parser.set_defaults(train='data/hog-train', test='data/hog-test')

if __name__ == '__main__':
	args = util.init(add_args)
	# Get data
	trainset = sae.dataset.SavedDataset(args.train)
	testset  = sae.dataset.SavedDataset(args.test)
	# Get model
	if args.load_path is not None:
		model = load_model(args.load_path)
	else:
		model = net.Net(arch=args.arch)

	# Greedy layer-wise training
	trainer.run(args, model.subnet(1),
		trainset=trainset, testset=testset,
		trainer_factory=Trainer)

	logging.info('Break in action')

	trainer.run(args, model.subnet(1),
		trainset=trainset, testset=testset,
		trainer_factory=Trainer)
