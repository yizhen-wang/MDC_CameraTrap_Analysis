import os
import json
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict
from datetime import datetime

import util as util

class Seq_Analyser:

	def __init__(self, time_interval=2):
		self.time_interval = time_interval

	def get_image_time(self, image_path):
		try:
			image = Image.open(image_path)
			exif_data = image._getexif()
			if exif_data is not None:
				for tag, value in exif_data.items():
					tag_name = TAGS.get(tag, tag)
					if tag_name == "DateTimeOriginal":
						return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
		except Exception as e:
			print(f"Can not load {image_path}'s taken time: {e}")
		return None

	def group_images_by_sequence(self, data_dir, output_dir=None, save_seq_info=True):
		if output_dir == None:
			output_dir = data_dir

		image_time_list = []
		valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
		for file_name in os.listdir(data_dir):
			if file_name.lower().endswith(valid_extensions):
				file_path = os.path.join(data_dir, file_name)
				time_taken = self.get_image_time(file_path)
				if time_taken:
					image_time_list.append((file_name, time_taken))

		image_time_list.sort(key=lambda x: x[1])

		sequences = defaultdict(lambda: {"image_list": [], "time_list": []})
		sequence_counter = 0
		prev_time = None

		for image_name, time_taken in image_time_list:
			if prev_time is None or (time_taken - prev_time).total_seconds() > self.time_interval:
				sequence_counter += 1
			sequence_name = f"sequence_{sequence_counter}"
			sequences[sequence_name]["image_list"].append(image_name)
			sequences[sequence_name]["time_list"].append(time_taken)
			prev_time = time_taken

		seq_info_dict = dict(sequences)

		if save_seq_info:
			seq_dict_out = os.path.join(output_dir, "seq_info_dict.json")
			with open(seq_dict_out, 'w') as json_file:
				json.dump(seq_info_dict, json_file, default=util.datetime_converter, indent=4)

		return seq_info_dict, len(image_time_list)




