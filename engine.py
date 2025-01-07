import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import util as util

class Animal_Detector:
	def __init__(self, data_dir, output_dir, animal_detector, animal_classifier, save_crops=True):
		if not os.path.exists(data_dir):
			raise ValueError(f"The directory '{data_dir}' does not exist.")

		self.data_dir = data_dir
		self.output_dir = output_dir
		self.animal_detector = animal_detector
		self.animal_classifier = animal_classifier
		self.save_crops = save_crops

	def run_analysis(self):
		print(f"Running detection on images in: {self.data_dir}")

		valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
		image_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
						if f.lower().endswith(valid_extensions)]
		print(f"Found {len(image_files)} image(s) in {self.data_dir}")

		analysis_results = {}

		crop_dir = os.path.join(self.output_dir, "CROP")
		if self.save_crops:
			os.makedirs(crop_dir, exist_ok=True)

		for img_path in tqdm(image_files):
			print(f"Processing image: {img_path}")

			# Run Animal Detection with MegaDetector
			analysis_results[img_path] = self.animal_detector.run_detection(img_path)

			# Run Animal Classification
			image = np.array(Image.open(img_path))
			cropped_images = []

			analysis_results[img_path]['cate'] = []
			analysis_results[img_path]['cla_conf'] = []

			for i in range(len(analysis_results[img_path]['bbox'])):
				if analysis_results[img_path]['super_cate'][i] == 1:
					analysis_results[img_path]['cate'].append('HUMAN')
					analysis_results[img_path]['cla_conf'].append(analysis_results[img_path]['det_conf'][i])
				elif analysis_results[img_path]['super_cate'][i] == 2:
					analysis_results[img_path]['cate'].append('VEHICLE')
					analysis_results[img_path]['cla_conf'].append(analysis_results[img_path]['det_conf'][i])
				elif analysis_results[img_path]['super_cate'][i] == 0:
					x1, y1, x2, y2 = analysis_results[img_path]['bbox'][i]
					crop = image[y1:y2, x1:x2]
					cropped_images.append(crop)

					# Save cropped image if save_crops is True
					if self.save_crops:
						crop_filename = os.path.join(crop_dir, 
							f"{os.path.basename(img_path).split('.')[0]}_crop_{i + 1}.png")
						Image.fromarray(crop).save(crop_filename)


					cla_result_dict = self.animal_classifier.run_classification(cropped_images)
					for key, value in cla_result_dict.items():
						if key in analysis_results[img_path]:
							analysis_results[img_path][key].extend(value)

			analysis_results[img_path]['super_cate'] = util.super_cate_trans(analysis_results[img_path]['super_cate'])

		util.save_analysis_result_json(analysis_results, self.output_dir)




class Animal_Detector_Seq:
	def __init__(self, data_dir, output_dir, animal_detector, animal_classifier, save_crops=True):
		if not os.path.exists(data_dir):
			raise ValueError(f"The directory '{data_dir}' does not exist.")

		self.data_dir = data_dir
		self.output_dir = output_dir
		self.animal_detector = animal_detector
		self.animal_classifier = animal_classifier
		self.save_crops = save_crops













