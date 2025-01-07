import os
import json

def get_parameter_number(net):
	total_num = sum(p.numel() for p in net.parameters())
	trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
	return {'Total': total_num, 'Trainable': trainable_num}

def super_cate_trans(super_cate_list):
	mapping = {0: "ANIMAL", 1: "HUMAN", 2: "VEHICLE"}
	return [mapping[val] for val in super_cate_list]


def save_analysis_result_json(result_dict, output_dir):
	output_file = os.path.join(output_dir, 'analysis_result.json')
	with open(output_file, "w") as f:
		json.dump(result_dict, f, indent=4)

