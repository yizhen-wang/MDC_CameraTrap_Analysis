import os
import json
import csv
from datetime import datetime

def get_parameter_number(net):
	total_num = sum(p.numel() for p in net.parameters())
	trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
	return {'Total': total_num, 'Trainable': trainable_num}

def super_cate_trans(super_cate_list):
	mapping = {0: "ANIMAL", 1: "HUMAN", 2: "VEHICLE"}
	return [mapping[val] for val in super_cate_list]

def datetime_converter(o):
	if isinstance(o, datetime):
		return o.strftime('%Y-%m-%d %H:%M:%S')
	raise TypeError("Type not serializable")

def save_analysis_result_json(result_dict, output_dir):
	output_file = os.path.join(output_dir, 'analysis_result.json')
	with open(output_file, "w") as f:
		json.dump(result_dict, f, default=datetime_converter, indent=4)

def save_analysis_result_csv(result_dict, output_dir):
	output_file = os.path.join(output_dir, 'analysis_result.csv')
	fieldnames = ['image_name'] + list(next(iter(result_dict.values())).keys())

	with open(output_file, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for image_name, data in result_dict.items():
			row = {'image_name': image_name}
			row.update(data)
			writer.writerow(row)

def save_analysis_result_seq_csv(result_dict, output_dir):
	output_file = os.path.join(output_dir, 'analysis_result.csv')
	fieldnames = ['seq_id'] + list(next(iter(result_dict.values())).keys())

	with open(output_file, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for seq_id, data in result_dict.items():
			row = {'seq_id': seq_id}
			row.update(data)
			writer.writerow(row)

