import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import logging
from logging import info

# Ref basic operations
# https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/
# Ref KLDivLoss
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

def assert_shape(t, shape):
	assert len(t.shape) == len(shape), (t is not None, t.shape)
	for i, v in enumerate(shape):
		assert t.shape[i] == v, t.shape

class Net(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.arch = None
		if 'state_dict' in kwargs:
			self.load_state_dict(kwargs['state_dict'])
		if 'arch' in kwargs:
			self._init_arch(kwargs['arch'])
	def _init_arch(self, arch):
		if self.arch is None:
			self.arch = arch
			dims = np.array(arch)
			dims = np.hstack((dims, dims[-2::-1]))
			self.dims = tuple(zip(dims[:-1], dims[1:]))
			self.layers = nn.Sequential(
				*[nn.Linear(*pair) for pair in self.dims]
				)
		elif not np.array_equal(self.arch, arch):
			raise Exception('Trying to set arch to a new shape after arch has already been set')
	def _build_block(self, index):
		n = len(self.dims)
		seq = nn.Sequential()
		seq.add_module('dropout', nn.Dropout(0.2))
		seq.add_module('linear', self.layers[index])
		if index not in [n-1, n//2-1]:
			seq.add_module('relu', nn.ReLU())
		return seq
	# Return an instance of nn.Sequential, containing n SEA's
	def subnet(self, n_saes):
		n = len(self.dims)
		info('Requested %d' % n_saes)
		if n_saes > n // 2:
			raise Exception('Requested %d SAEs, but only %d are present' % (n_saes, len(self.arch)))
		seq = nn.Sequential()
		seq.arch = self.arch
		# Create encoder blocks
		for i in range(n_saes):
			seq.add_module(str(i), self._build_block(i))
		# Create decoder blocks
		for i in range(n-n_saes, n):
			seq.add_module(str(i), self._build_block(i))
		return seq
	# Return layers only up to the bottleneck, enclose in nn.Sequential
	def get_encoder(self):
		seq = nn.Sequential()
		# Create encoder blocks
		for i in range(len(self.dims) // 2):
			seq.add_module(str(i), self._build_block(i))
		return seq

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(message)s')
	info('Running...')
	net = Net(arch=[16, 8, 4, 2])
	info(net.arch)
	info(net.dims)
	info(net.subnet(1))
	info(net.subnet(2))
	info(net.subnet(3))
