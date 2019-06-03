import sys, os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import data.stl10_input as stl

class Dataset(data.Dataset):
	def __getitem__(self, index):
		return {
			'X': self.images[index],
			'y': self.labels[index],
		}

	def __len__(self):
		return len(self.images)

	def save(self, savedir):
		os.makedirs(savedir, exist_ok=True)
		torch.save(self.images, os.path.join(savedir, 'images.pth'))
		torch.save(self.labels, os.path.join(savedir, 'labels.pth'))

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
		im_arry = stl.read_all_images(stl.DATA_PATH)
		im_list = [ cv2.resize(im, dsize=(224,224), interpolation=cv2.INTER_CUBIC) for im in im_arry ]
		self.images = torch.from_numpy( np.stack(im_list) ).float().permute(0,3,1,2)
		self.labels = torch.from_numpy( stl.read_labels(stl.LABEL_PATH) ) - 1

class SplitDataset(Dataset):
	def __init__(self, images, labels, indices):
		super().__init__()
		self.images = images[indices]
		self.labels = labels[indices]

class SavedDataset(Dataset):
	def __init__(self, savedir):
		super().__init__()
		self.images = torch.load(os.path.join(savedir, 'images.pth'))
		self.labels = torch.load(os.path.join(savedir, 'labels.pth'))

if __name__ == '__main__':
	starter = StarterDataset()
	traiset, testset = starter.split(0.1)
	traiset.save('data/train')
	testset.save('data/test')
	print('Saved trainset of len %d' % len(traiset))
	print('Saved testset  of len %d' % len(testset))
	SavedDataset('data/train')
	SavedDataset('data/test')
