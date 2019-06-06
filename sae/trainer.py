import os, torch, logging
from . import net
from shared import util
from shared import trainer as generic_trainer
import sae.dataset

# Use this class as a trainer_factory for the run function. Or subclass this class (for pretraining, which requires a different loss)
class Trainer(generic_trainer.Trainer):
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

class Runner(object):
	def __init__(self, trainer_factory):
		self.trainer_factory = trainer_factory
		self.args = util.init(add_args)
		# Get data
		self.trainset = sae.dataset.SavedDataset(self.args.train)
		self.testset  = sae.dataset.SavedDataset(self.args.test)
		# Get model
		if self.args.load_path is not None:
			self.model = load_model(self.args.load_path)
		else:
			self.model = net.Net(arch=self.args.arch)

	def run(self):
		generic_trainer.run(self.args, self.model,
			trainset=self.trainset, testset=self.testset,
			trainer_factory=self.trainer_factory)

def load_model(load_path):
	dump = torch.load(load_path)
	model = net.Net(arch=dump['arch'])
	model.load_state_dict( dump['state_dict'] )
	return model

def add_args(parser):
	parser.add_argument('--arch', default=[1296,500,500,2000,10], nargs='+', type=int, help='Dimensions for stacked autoencoders')
	parser.set_defaults(train='data/hog-train', test='data/hog-test')
