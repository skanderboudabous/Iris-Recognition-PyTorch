#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import numpy as np
from glob import glob
import os, cv2, torch

from base import BaseDataset
from dataloader import transforms
from dataloader.augmentation import Augmentation


#------------------------------------------------------------------------------
#	TxtDataset
#------------------------------------------------------------------------------
class TxtDataset(BaseDataset):

	def __init__(self, txtfile, num_classes, kfolds=1, use_sigmoid=True, color_channel="RGB",
		input_size=(256, 256), normalize=True, one_hot=False, is_training=True,
		rot90=0.3, flip_hor=0.5, flip_ver=0.5, brightness=0.2, contrast=0.1, shift=0.1625, scale=0.6, rotate=10,
		img_loader_mode='pillow', normalize_mode='imagenet'):

		# Init BaseDataset
		super(TxtDataset, self).__init__(img_loader_mode=img_loader_mode, normalize_mode=normalize_mode)

		# Data augmentation
		self.augmentor = Augmentation(
			rot90=rot90, flip_hor=flip_hor, flip_ver=flip_ver, brightness=brightness,
			contrast=contrast, shift=shift, scale=scale, rotate=rotate,
		)

		# Get image files
		self._get_image_files(txtfile)
		self._get_labels()

		# K-fold cross validation
		self.kfolds = kfolds
		if kfolds!=1:
			self.folds = self.split_kfolds(self.image_files, self.labels, kfolds)
			print("[{}] Split into {} cross-validation folds: {}".format(
				self.__class__.__name__, kfolds, [len(fold) for fold in self.folds]
			))

		# Parameters
		self.use_sigmoid = use_sigmoid
		self.num_classes = num_classes+1 if self.use_sigmoid else num_classes
		self.color_channel = color_channel
		self.input_size = tuple(input_size)
		self.is_training = is_training
		self.normalize = normalize
		self.one_hot = one_hot

	def __len__(self):
		if self.kfolds==1:
			return len(self.image_files)
		else:
			if not hasattr(self, 'fold_image_files'):
				raise ValueError("Please call `set_fold` function before creating dataloader")
			else:
				return len(self.fold_image_files)

	def __getitem__(self, idx):
		# Read image and label
		image = self.img_loader(self.image_files[idx])
		label = np.array(self.labels[idx])

		# Data augmentation
		if self.is_training:
			image = self._augment_data(image)

		# Preprocess image
		image = transforms.resize_image(image, height=self.input_size[0], width=self.input_size[1], mode=cv2.INTER_LINEAR)
		if self.normalize:
			image = self.normalize_fnc(image)
		image = np.transpose(image, axes=(2,0,1))

		# Preprocess label
		if self.one_hot:
			label = (np.arange(self.num_classes) == label[..., None])

		# Convert to tensor and return
		data = {
			"images": torch.from_numpy(image.astype(np.float32)),
			"targets": torch.from_numpy(label.astype(np.int64)),
		}
		return data

	def set_fold(self, fold_idx):
		assert 0 <= fold_idx < self.kfolds
		fold = self.folds[fold_idx]
		self.fold_image_files = [self.image_files[idx] for idx in fold]
		self.fold_labels = [self.labels[idx] for idx in fold]
		print("[%s] Set fold-%d with %d samples" % (
			self.__class__.__name__, fold_idx, len(self.fold_image_files)
		))

	def _get_image_files(self, txtfile):
		fp = open(txtfile, 'r')
		image_files = [line for line in fp.read().split("\n") if len(line)]
		self.image_files = self.filter_files(image_files)
		print("[%s] Number of samples:" % (self.__class__.__name__), len(self.image_files))
		self.check_filepaths(self.image_files)

	def _get_labels(self):
		self.labels = []
		raise NotImplementedError

	def _augment_data(self, image):
		image = self.augmentor(image)
		return image


#------------------------------------------------------------------------------
#  MMU2Dataset
#------------------------------------------------------------------------------
class MMU2Dataset(TxtDataset):
	def __init__(self, **kargs):
		super(MMU2Dataset, self).__init__(**kargs)

	def _get_labels(self):
		self.labels = [
			int(os.path.basename(file).split(".")[0][:-4])-1
			for file in self.image_files
		]


#------------------------------------------------------------------------------
#  CASIA1Dataset
#------------------------------------------------------------------------------
class CASIA1Dataset(TxtDataset):
	def __init__(self, **kargs):
		super(CASIA1Dataset, self).__init__(**kargs)

	def _get_labels(self):
		self.labels = [
			int(os.path.basename(file).split('_')[0]) - 1
			for file in self.image_files
		]


#------------------------------------------------------------------------------
#  CASIA4ThousandDataset
#------------------------------------------------------------------------------
class CASIA4ThousandDataset(TxtDataset):
	def __init__(self, **kargs):
		super(CASIA4ThousandDataset, self).__init__(**kargs)

	def _get_labels(self):
		self.labels = [
			int(os.path.basename(file)[2:5])
			for file in self.image_files
		]


#------------------------------------------------------------------------------
#  CASIA4IntervalDataset
#------------------------------------------------------------------------------
class CASIA4IntervalDataset(TxtDataset):
	def __init__(self, **kargs):
		super(CASIA4IntervalDataset, self).__init__(**kargs)

	def _get_labels(self):
		self.labels = [
			int(os.path.basename(file)[2:5]) - 1
			for file in self.image_files
		]
