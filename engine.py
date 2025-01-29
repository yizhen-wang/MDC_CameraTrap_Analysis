import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import util as util
import seq_analysis as seq_ana
import detection as det
import classification as cla

class Animal_Detector:
	def __init__(self, data_dir, device, save_crops=False, **kwargs):
		if not os.path.exists(data_dir):
			raise ValueError(f"The directory '{data_dir}' does not exist.")

		self.data_dir = data_dir
		self.output_dir = kwargs.get('output_dir')
		self.animal_detector = det.Animal_Detector(device, conf_threshold=kwargs.get('conf_threshold'))
		self.disable_classification = kwargs.get('disable_classification')
		if not self.disable_classification:
			self.animal_classifier = cla.Animal_Classifier(device)
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

		for image_path in tqdm(image_files):
			print(f"Processing image: {image_path}")
			image_name = os.path.basename(image_path)

			# Run Animal Detection with MegaDetector
			analysis_results[image_name] = {}
			analysis_results[image_name]['image_path'] = image_path
			det_result_dict = self.animal_detector.run_detection(image_path)
			for key, value in det_result_dict.items():
				analysis_results[image_name][key] = value

			analysis_results[image_name]['super_cate'] = util.super_cate_trans(analysis_results[image_name]['super_cate'])
			
			if not self.disable_classification:
				# Run Animal Classification
				image = np.array(Image.open(image_path))
				cropped_images = []

				analysis_results[image_name]['cate'] = []
				analysis_results[image_name]['cla_conf'] = []

				for i in range(len(analysis_results[image_name]['bbox'])):
					if analysis_results[image_name]['super_cate'][i] == 'HUMAN':
						analysis_results[image_name]['cate'].append('HUMAN')
						analysis_results[image_name]['cla_conf'].append(analysis_results[image_name]['det_conf'][i])
					elif analysis_results[image_name]['super_cate'][i] == 'VEHICLE':
						analysis_results[image_name]['cate'].append('VEHICLE')
						analysis_results[image_name]['cla_conf'].append(analysis_results[image_name]['det_conf'][i])
					elif analysis_results[image_name]['super_cate'][i] == 'ANIMAL':
						x1, y1, x2, y2 = analysis_results[image_name]['bbox'][i]
						crop = image[y1:y2, x1:x2]
						cropped_images.append(crop)

						# Save cropped image if save_crops is True
						if self.save_crops:
							crop_filename = os.path.join(crop_dir, 
								f"{os.path.basename(image_path).split('.')[0]}_crop_{i + 1}.png")
							Image.fromarray(crop).save(crop_filename)

				cla_result_dict = self.animal_classifier.run_classification(cropped_images)
				for key, value in cla_result_dict.items():
					if key in analysis_results[image_name]:
						analysis_results[image_name][key].extend(value)

		analysis_results = self.format_output_dict(analysis_results)
		util.save_analysis_result_json(analysis_results, self.output_dir)
		util.save_analysis_result_csv(analysis_results, self.output_dir)

	def format_output_dict(self, analysis_results):
		analysis_results_reformat = {}
		for image_name in analysis_results.keys():
			analysis_results_reformat[image_name] = {}
			analysis_results_reformat[image_name]['image_path'] = analysis_results[image_name]['image_path']
			analysis_results_reformat[image_name]['bbox'] = analysis_results[image_name]['bbox']
			analysis_results_reformat[image_name]['det_conf'] = [round(conf, 4) for conf in 
																	analysis_results[image_name]['det_conf']]

			if len(analysis_results_reformat[image_name]['bbox']) > 0:
				max_det_index = analysis_results[image_name]['det_conf'].index(max(analysis_results[image_name]['det_conf']))
				analysis_results_reformat[image_name]['super_cate'] = analysis_results[image_name]['super_cate'][max_det_index]
			else:
				analysis_results_reformat[image_name]['super_cate'] = 'None'

			if analysis_results_reformat[image_name]['super_cate'] == 'ANIMAL':
				analysis_results_reformat[image_name]['animal_exist'] = 'True'
			else:
				analysis_results_reformat[image_name]['animal_exist'] = 'False'

			if not self.disable_classification:
				if len(analysis_results_reformat[image_name]['bbox']) > 0 and analysis_results_reformat[image_name]['super_cate'] == 'ANIMAL':
					analysis_results_reformat[image_name]['cla_conf'] = round(max(analysis_results[image_name]['cla_conf']),4)
					max_cla_index = analysis_results[image_name]['cla_conf'].index(max(analysis_results[image_name]['cla_conf']))
					analysis_results_reformat[image_name]['category'] = analysis_results[image_name]['cate'][max_det_index]
				else:
					analysis_results_reformat[image_name]['cla_conf'] = 0
					analysis_results_reformat[image_name]['category'] = 'None'

				animal_count = 0
				for i in range(len(analysis_results[image_name]['bbox'])):
					if analysis_results[image_name]['super_cate'][i] == 'ANIMAL' and analysis_results_reformat[image_name]['category'] != 'NO_ANIMAL':
						animal_count+=1
				analysis_results_reformat[image_name]['animal_count'] = animal_count
			else:
				animal_count = 0
				for i in range(len(analysis_results[image_name]['bbox'])):
					if analysis_results[image_name]['super_cate'][i] == 'ANIMAL':
						animal_count+=1
				analysis_results_reformat[image_name]['animal_count'] = animal_count

		return analysis_results_reformat



class Animal_Detector_Seq:
	def __init__(self, data_dir, device, save_crops=False, include_image_level_res=True, **kwargs):
		if not os.path.exists(data_dir):
			raise ValueError(f"The directory '{data_dir}' does not exist.")

		self.data_dir = data_dir
		self.output_dir = kwargs.get('output_dir')
		self.animal_detector = det.Animal_Detector(device, conf_threshold=kwargs.get('conf_threshold'))
		self.disable_classification = kwargs.get('disable_classification')
		if not self.disable_classification:
			self.animal_classifier = cla.Animal_Classifier(device)
		self.save_crops = save_crops
		self.include_image_level_res = include_image_level_res

	def run_analysis(self):
		print("Sequential Analysis")
		seq_analyser = seq_ana.Seq_Analyser()
		seq_info_dict, image_count = seq_analyser.group_images_by_sequence(self.data_dir, self.output_dir)

		print(f"Found {image_count} image(s) in {self.data_dir}")
		print(f"Found {len(seq_info_dict.keys())} sequence(s) in {self.data_dir}")

		analysis_results = {}
		crop_dir = os.path.join(self.output_dir, "CROP")
		if self.save_crops:
			os.makedirs(crop_dir, exist_ok=True)

		for seq_id in tqdm(seq_info_dict.keys()):
			print(f"Processing Sequence: {seq_id}")
			analysis_results[seq_id] = {}
			analysis_results[seq_id]['image_list'] = seq_info_dict[seq_id]['image_list']
			analysis_results[seq_id]['time_list'] = seq_info_dict[seq_id]['time_list']

			analysis_results[seq_id]['det_result'] = []
			if not self.disable_classification:
				analysis_results[seq_id]['cla_result'] = []

			for image_name in analysis_results[seq_id]['image_list']:
				# Run Animal Detection with MegaDetector
				image_path = os.path.join(self.data_dir, image_name)
				det_result_dict = self.animal_detector.run_detection(image_path)
				det_result_dict['super_cate'] = util.super_cate_trans(det_result_dict['super_cate'])
				analysis_results[seq_id]['det_result'].append(det_result_dict)

				if not self.disable_classification:
					# Run Animal Classification
					image = np.array(Image.open(image_path))
					cropped_images = []

					cla_result_dict = {}
					cla_result_dict['cate'] = []
					cla_result_dict['cla_conf'] = []

					for i in range(len(det_result_dict['bbox'])):
						if det_result_dict['super_cate'][i] == 'HUMAN':
							cla_result_dict['cate'].append('HUMAN')
							cla_result_dict['cla_conf'].append(det_result_dict['det_conf'][i])
						elif det_result_dict['super_cate'][i] == 'VEHICLE':
							cla_result_dict['cate'].append('VEHICLE')
							cla_result_dict['cla_conf'].append(det_result_dict['det_conf'][i])
						elif det_result_dict['super_cate'][i] == 'ANIMAL':
							x1, y1, x2, y2 = det_result_dict['bbox'][i]
							crop = image[y1:y2, x1:x2]
							cropped_images.append(crop)

							# Save cropped image if save_crops is True
							if self.save_crops:
								crop_filename = os.path.join(crop_dir, 
									f"{os.path.basename(image_path).split('.')[0]}_crop_{i + 1}.png")
								Image.fromarray(crop).save(crop_filename)

					cla_result = self.animal_classifier.run_classification(cropped_images)
					for key, value in cla_result.items():
						if key in cla_result_dict.keys():
							cla_result_dict[key].extend(value)

					analysis_results[seq_id]['cla_result'].append(cla_result_dict)

		if self.include_image_level_res:
			analysis_results_image = self.format_output_dict_to_image(analysis_results)
			util.save_analysis_result_json(analysis_results_image, self.output_dir, 'analysis_result_image.json')
			util.save_analysis_result_csv(analysis_results_image, self.output_dir, 'analysis_result_image.csv')

		analysis_results = self.format_output_dict(analysis_results)
		util.save_analysis_result_json(analysis_results, self.output_dir)
		util.save_analysis_result_seq_csv(analysis_results, self.output_dir)

	def format_output_dict(self, analysis_results):
		analysis_results_reformat = {}
		for seq_id in tqdm(analysis_results.keys()):
			analysis_results_reformat[seq_id] = {}
			analysis_results_reformat[seq_id]['image_list'] = analysis_results[seq_id]['image_list']
			analysis_results_reformat[seq_id]['time_list'] = analysis_results[seq_id]['time_list']

			analysis_results_reformat[seq_id]['animal_exist'] = 'False'
			analysis_results_reformat[seq_id]['animal_count'] = 0

			if not self.disable_classification:
				analysis_results_reformat[seq_id]['category'] = ''
				analysis_results_reformat[seq_id]['cla_conf'] = 0

			for img_idx in range(len(analysis_results[seq_id]['image_list'])):
				if len(analysis_results[seq_id]['det_result'][img_idx]['bbox']) == 0:
					continue
				elif self.disable_classification:
					animal_count = 0
					for det_idx in range(len(analysis_results[seq_id]['det_result'][img_idx]['bbox'])):
						if analysis_results[seq_id]['det_result'][img_idx]['super_cate'][det_idx] == 'ANIMAL':
							animal_count += 1

					if animal_count > analysis_results_reformat[seq_id]['animal_count']:
						analysis_results_reformat[seq_id]['animal_count'] = animal_count
				else:
					animal_count = 0
					for det_idx in range(len(analysis_results[seq_id]['det_result'][img_idx]['bbox'])):
						if (
							analysis_results[seq_id]['det_result'][img_idx]['super_cate'][det_idx] == 'ANIMAL' 
							and analysis_results[seq_id]['cla_result'][img_idx]['cate'][det_idx] != 'NO_ANIMAL'
							):
							animal_count += 1

						if analysis_results[seq_id]['cla_result'][img_idx]['cate'][det_idx] != 'NO_ANIMAL':
							if analysis_results[seq_id]['cla_result'][img_idx]['cla_conf'][det_idx] > analysis_results_reformat[seq_id]['cla_conf']:
								analysis_results_reformat[seq_id]['cla_conf'] = round(analysis_results[seq_id]['cla_result'][img_idx]['cla_conf'][det_idx],4)
								analysis_results_reformat[seq_id]['category'] = analysis_results[seq_id]['cla_result'][img_idx]['cate'][det_idx]

					if animal_count > analysis_results_reformat[seq_id]['animal_count']:
						analysis_results_reformat[seq_id]['animal_count'] = animal_count

			if analysis_results_reformat[seq_id]['animal_count']>0:
				analysis_results_reformat[seq_id]['animal_exist'] = 'True'

		return analysis_results_reformat

	def format_output_dict_to_image(self, analysis_results):
		analysis_results_reformat = {}
		for seq_id in tqdm(analysis_results.keys()):
			for img_idx, image_name in enumerate(analysis_results[seq_id]['image_list']):
				analysis_results_reformat[image_name] = {}
				analysis_results_reformat[image_name]['image_path'] = os.path.join(self.data_dir, image_name)
				analysis_results_reformat[image_name]['bbox'] = analysis_results[seq_id]['det_result'][img_idx]['bbox']

				if len(analysis_results[seq_id]['det_result'][img_idx]['bbox']) > 0:
					max_det_index = analysis_results[seq_id]['det_result'][img_idx]['bbox'].index(
						max(analysis_results[seq_id]['det_result'][img_idx]['bbox']))
					analysis_results_reformat[image_name]['super_cate'] = analysis_results[seq_id]['det_result'][img_idx]['super_cate'][max_det_index]
				else:
					analysis_results_reformat[image_name]['super_cate'] = 'None'

				if analysis_results_reformat[image_name]['super_cate'] == 'ANIMAL':
					analysis_results_reformat[image_name]['animal_exist'] = 'True'
				else:
					analysis_results_reformat[image_name]['animal_exist'] = 'False'

				if not self.disable_classification:
					if len(analysis_results_reformat[image_name]['bbox']) > 0 and analysis_results_reformat[image_name]['super_cate'] == 'ANIMAL':
						analysis_results_reformat[image_name]['cla_conf'] = round(max(analysis_results[seq_id]['cla_result'][img_idx]['cla_conf']),4)
						max_cla_index = analysis_results[image_name]['cla_conf'].index(max(analysis_results[seq_id]['cla_result'][img_idx]['cla_conf']))
						analysis_results_reformat[image_name]['category'] = analysis_results[seq_id]['cla_result'][img_idx][max_det_index]
					else:
						analysis_results_reformat[image_name]['cla_conf'] = 0
						analysis_results_reformat[image_name]['category'] = 'None'
				else:
					analysis_results_reformat[image_name]['animal_count'] = 0
					for i in range(len(analysis_results[seq_id]['det_result'][img_idx]['bbox'])):
						if analysis_results[seq_id]['det_result'][img_idx]['super_cate'][i] == 'ANIMAL':
							analysis_results_reformat[image_name]['animal_count']+=1

		return analysis_results_reformat


					














