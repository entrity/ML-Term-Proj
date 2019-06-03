import sys, os
import numpy as np
import torch
import torch.utils.data as data
import data.stl10_input as stl

class Dataset(data.Dataset):
	def __getitem__(self, index):
		return {
			X: self.images[index],
			y: self.labels[index],
		}

	def __len__(self):
		return len(self.images)

	# Format should be a filename string like "train-%s.pth"
	def save(self, format):
		torch.save(self.images, 'data/'+format % 'images')
		torch.save(self.labels, 'data/'+format % 'labels')

	# Returns 2 new datasets: train and test
	def split(self, test_set_ratio):
		n = len(self.images)
		k = round(n * test_set_ratio)
		idxs = torch.randperm(n)
		test_idxs = idxs[:k]
		trai_idxs = idxs[k:]
		traiset = SplitDataset(self.images, self.labels, trai_idxs)
		testset = SplitDataset(self.images, self.labels, test_idxs)
		return traiset, testset

class StarterDataset(Dataset):
	def __init__(self):
		super().__init__()
		convert = lambda nparray : torch.from_numpy(nparray)
		self.images = convert(stl.read_all_images(stl.DATA_PATH)).float()
		self.labels = convert(stl.read_labels(stl.LABEL_PATH)) - 1

class SplitDataset(Dataset):
	def __init__(self, images, labels, indices):
		super().__init__()
		self.images = images[indices]
		self.labels = labels[indices]

class SavedDataset(Dataset):
	def __init__(self, filename_format):
		super().__init__()
		load = lambda mode : torch.load('data/' + filename_format % mode)
		self.images = load('images')
		self.labels = load('labels')

if __name__ == '__main__':
	starter = StarterDataset()
	traiset, testset = starter.split(0.1)
	traiset.save('train-%s.pth')
	testset.save('test-%s.pth')
	print('Saved trainset of len %d' % len(traiset))
	print('Saved testset  of len %d' % len(testset))
	SavedDataset('train-%s.pth')
	SavedDataset('test-%s.pth')
