import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import util as util

class Animal_Classifier:
	def __init__(self, device):
		self.device = device
		self.category_list = ["ARMADILLO","BIRD","BOBCAT","COYOTE","DOG","FERAL_HOG","FOX","HOUSE_CAT",
							  "MOUSE","OPOSSUM","RACCOON","SQUIRREL","WHITE_TAIL_DEER","NO_ANIMAL"]
		self.model_path = os.path.join("..", "MODEL", "effnet_b3.pt")
		self.classification_model = self._load_model(self.device)
		self.cla_transforms = self._get_transforms()

	def _load_model(self, device):
		model = EfficientNet.from_pretrained('efficientnet-b3')
		IN_FEATURES = model._fc.in_features
		OUTPUT_DIM = len(self.category_list)
		model._fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

		if device == 'cpu':
			model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
		elif device == 'cuda':
			model.load_state_dict(torch.load(self.model_path))
		else:
			raise ValueError(f"Unsupported device: {device}")
		return model

	def _get_transforms(self):
		pretrained_size  = 224
		pretrained_means = [0.485, 0.456, 0.406]
		pretrained_stds  = [0.229, 0.224, 0.225]

		cla_transforms = transforms.Compose([
							transforms.ToTensor(),
							transforms.Resize(pretrained_size),
							transforms.CenterCrop(pretrained_size),
							transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
							])
		return cla_transforms

	def run_classification(self, crop_images):
		cla_result = {}
		cla_result['cate'] = []
		cla_result['cla_conf'] = []

		for crop in crop_images:
			x = self.cla_transforms(crop)
			x = x.to(self.device)
			x = x.unsqueeze(0)
			y_pred = self.classification_model(x)

			pred, conf = self.process_model_output(y_pred)
			cla_result['cate'].append(pred)
			cla_result['cla_conf'].append(conf)
		return cla_result

	def process_model_output(self, y_pred):
		"""
			Process the raw output of the model to obtain probabilities and corresponding class names.

			Returns:
				- prediction (str): The predicted class name with the highest confidence.
				- confidence (float): The confidence (probability) of the predicted class.
		"""

		# Apply softmax to convert scores to probabilities
		probabilities = F.softmax(y_pred, dim=1).squeeze()

		# Find the index of the highest probability
		top_index = torch.argmax(probabilities).item()

		# Get the predicted class name and confidence
		prediction = self.category_list[top_index]
		confidence = probabilities[top_index].item()

		# Create a list of (class_name, probability) tuples
		# prob_list = [(self.category_list[i], probabilities[i].item()) for i in range(len(self.category_list))]

		return prediction, confidence










