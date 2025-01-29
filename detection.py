import os
import torch
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
import util as util

class Animal_Detector:
	def __init__(self, device, conf_threshold=0.5, scale=1.0):
		"""
		Initialize the Detector class.
		"""
		self.conf_threshold = conf_threshold
		print(f"Using confidence threshold: {self.conf_threshold}")
		self.detection_model = self._load_model(device)
		self.scale = scale

	def _load_model(self, device):
		model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="yolov9c")
		return model

	def resize_image(self, img_path):
		if self.scale <= 0:
			raise ValueError("Scale factor must be greater than 0.")

		#img_pil = Image.open(img_path)
		#new_width = int(img_pil.width * self.scale)
		#new_height = int(img_pil.height * self.scale)
		#img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)

		image = np.array(Image.open(img_path))
		new_width = int(image.shape[1] * self.scale)
		new_height = int(image.shape[0] * self.scale)
		img_resized = cv2.resize(image, (new_width, new_height))

		#image = np.array(Image.open(img_path))
		#h, w = image.shape[:2]
		#factor = round(1 / self.scale)
		#img_resized = image[::factor, ::factor, :]

		return np.array(img_resized)

	def run_detection(self, img_path):
		"""
		Run the detection process.
		"""
		if self.scale < 1:
			image = self.resize_image(img_path)
			print(image.shape)
		else:
			image = np.array(Image.open(img_path))
			print(image.shape)

		det_result = self.detection_model.single_image_detection(image)
		det_result = self._format_det_result(det_result, categories=self.detection_model.CLASS_NAMES)
		return det_result

	def _format_det_result(self, det_result, categories=None, exclude_category_ids=[]):
		anno_dict = {}
		
		bboxes = det_result["detections"].xyxy.astype(int)
		category = det_result["detections"].class_id
		confidences = det_result["detections"].confidence

		valid_category_mask = ~np.isin(category, exclude_category_ids)
		valid_conf_mask = confidences >= self.conf_threshold
		valid_mask = valid_category_mask & valid_conf_mask

		anno_dict['bbox'] = bboxes[valid_mask].tolist()
		anno_dict['det_conf'] = confidences[valid_mask].tolist()
		anno_dict['super_cate'] = category[valid_mask].tolist()

		return anno_dict









