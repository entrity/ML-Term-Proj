import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
import sys, os, logging, datetime
from collections import deque
import shared.kl_loss as kl_loss
import data.dataset

class Trainer(object):
	def __init__(self, trainloader, testloader, net, optim, state, **kwargs):
		self.trainloader = trainloader
		self.testloader = testloader
		self.net = net
		self.optim = optim
		self.scheduler = kwargs.get('scheduler', None)
		self.loss_fn = kl_loss.compute_kl_div_loss
		self.test_every = kwargs.get('test_every', 100)
		self.print_every = kwargs.get('print_every', 100)
		self.save_every  = kwargs.get('save_every', 1)
		self.save_path   = kwargs['save_path']
		self.k_clusters  = kwargs.get('k_clusters', 10)
		self._tics = []

	def run(self, n_epochs, epoch_i=0, best_test=999999):
		self.net.train()
		self.iter_i       = 0
		self.best_test    = best_test
		self.test_losses  = []
		self.train_losses = []
		self.batch_losses = deque(maxlen=1000)
		for self.epoch_i in range(epoch_i, n_epochs):
			self.train_epoch()
			if self.epoch_i % self.save_every == 0:
				self.save_model('epoch_end')
		
	def train_epoch(self):
		self.tic()
		for self.batch_i, batch in enumerate(self.trainloader):
			self.train_batch(batch)
			if self.test_every and self.iter_i % self.test_every and self.testloader is not None:
				self.test()
			self.iter_i += 1
		self.toc()
		self.log_epoch_loss()
		if self.testloader is not None: self.test()
		self.epoch_i += 1

	def train_batch(self, batch):
		self.tic()
		self.optim.zero_grad()
		loss = self._loss(batch)
		loss.backward()
		self.optim.step()
		self.batch_losses.append(loss.item())
		tictoc = self.toc()
		if self.print_every and self.iter_i % self.print_every == 0:
			self.print('TRAIN', loss.item(), tictoc)

	def _loss(self, batch):
		X = batch['X'].cuda()
		embedding = self.net(X)
		loss = self.loss_fn(embedding, self.k_clusters)

	def _loss_for_dataloader(self, dataloader, mode):
		self.tic()
		self.net.eval()
		self.net.zero_grad()
		sizes  = np.array([len(batch) for batch in dataloader])
		losses = np.array([self._loss(batch).item() for batch in dataloader])
		loss   = np.mean(losses / sizes)
		self.net.train()
		tictoc = self.toc()
		self.print(mode, loss, tictoc)
		return loss

	def test(self):
		loss = self._loss_for_dataloader(self.testloader, 'TEST')
		self.test_losses.append(loss)
		if loss < self.best_test:
			self.best_test = loss
			self.save_model()
		if self.scheduler is not None:
			self.scheduler.step(loss)

	def log_epoch_loss(self):
		loss = self._loss_for_dataloader(self.trainloader, 'EPOCH')
		self.train_losses.append(loss)

	def tic(self):
		self._tics.append(datetime.datetime.now())
	
	def toc(self):
		return (datetime.datetime.now() - self._tics.pop())

	def print(self, mode, loss, tictoc):
		logging.info('%6s  %4d:%-7d %e %s' % (mode, self.epoch_i, self.iter_i, loss, str(tictoc)))

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
			'iter_i': self.iter_i,
			'epoch_i': self.epoch_i,
			'batch_i': self.batch_i,
			}, save_path)

def run(args, net, **kwargs):
	net.cuda()
	# Make dirs
	os.makedirs(os.path.dirname( args.save_path ), exist_ok=True)
	os.makedirs(os.path.dirname( args.log_path ), exist_ok=True)
	# Load datasets
	trainset = kwargs['trainset'] if 'trainset' in kwargs else data.dataset.SavedDataset(args.train)
	testset  = kwargs['testset']  if 'testset'  in kwargs else data.dataset.SavedDataset(args.test)
	# Make dataloaders
	trainloader = DataLoader( trainset, batch_size=args.train_bs, shuffle=True )
	testloader  = DataLoader( testset,  batch_size=args.test_bs )
	# Build optimizer
	optim = torch.optim.SGD( net.parameters(), lr=args.lr )
	# Load from dump
	if args.do_continue:
		state = torch.load( args.load_path )
		optim.load_state_dict( state['optim_state_dict'] )
	else:
		state = {}
	# Build trainer
	trainer = Trainer(trainloader, testloader, net, optim, state,
		save_path=args.save_path,
		test_every=args.test_every, print_every=args.print_every, save_every=args.save_every)
	# Train
	trainer.run(args.ep, state.get('epoch_i', 0), state.get('best_test', 9999999))
