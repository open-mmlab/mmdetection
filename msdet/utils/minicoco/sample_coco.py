import json
import argparse
from random import shuffle

from dataloader import CocoDataset
from sampler_utils import get_coco_object_size_info, get_coco_class_object_counts
from tqdm import tqdm


# default area ranges defined in coco
areaRng = [32 ** 2, 96 ** 2, 1e5 ** 2]

def sampling(keys, dataset_train, size_dict, parser, num_count):
	# now sample!!
	imgs_best_sample = {}
	ratio_list = []
	best_diff = 1_000_000

	for rr in tqdm(range(parser.run_count)):
		imgs = {}

		# shuffle keys
		shuffle(keys)

		# select first N images
		for i in keys[:num_count]:
			imgs[i] = dataset_train.coco.imgToAnns[i]

		# now check for category based annotations
		# annot_sampled = np.zeros(90, int)
		annot_sampled = {}
		for k, v in imgs.items():
			for it in v:
				area = it['bbox'][2] * it['bbox'][3]
				cat = it['category_id']
				if area < areaRng[0]:
					kk = str(cat) + "_S"
				elif area < areaRng[1]:
					kk = str(cat) + "_M"
				else:
					kk = str(cat) + "_L"

				if kk in annot_sampled:
					annot_sampled[kk] += 1
				else:
					annot_sampled[kk] = 1
     
		# calculate ratios
		ratios_obj_count = {}
		# ratios_obj_size = {}

		failed_run = False
		for k, v in size_dict.items():
			if not k in annot_sampled:
				failed_run = True
				print(k)
				break
			ratios_obj_count[k] = annot_sampled[k] / float(v)
   
		if failed_run:
			continue

		ratio_list.append(ratios_obj_count)

		min_ratio = min(ratios_obj_count.values())
		max_ratio = max(ratios_obj_count.values())

		diff = max_ratio - min_ratio

		if diff < best_diff:
			best_diff = diff
			imgs_best_sample = imgs
			print(best_diff)

	return imgs_best_sample


def main(args=None):
	parser = argparse.ArgumentParser(description='Mini COCO Sampling Script')
	parser.add_argument('--coco_path', help='Path to COCO directory', default="/data/sung/dataset/coco")
	parser.add_argument('--save_file_name_train', help='Save file name', default="instances_train2017_minicoco")
	parser.add_argument('--save_file_name_val', help='Save file name', default="instances_val2017_minicoco")
	parser.add_argument('--sample_image_count_train', help='How many images you want to sample', type=int, default=40000)
	parser.add_argument('--sample_image_count_val', help='How many images you want to sample', type=int, default=10000)
	parser.add_argument('--run_count', help='How many times you want to run sampling', type=int, default=1000)
	parser.add_argument('--debug', help='Print useful info', action='store_true')
	parser = parser.parse_args(args)

	dataset_train = CocoDataset(parser.coco_path, set_name='train2017')

	# get coco class based object counts
	annot_dict = get_coco_class_object_counts(dataset_train)
	if parser.debug:
		print(f"COCO object counts in each class:\n{annot_dict}")

	# here extract object sizes.
	size_dict = get_coco_object_size_info(dataset_train)

	keys = []
	# get all keys in coco train set, total image count!
	for k, v in dataset_train.coco.imgToAnns.items():
		keys.append(k)

	imgs_best_sample_train = sampling(keys, dataset_train, size_dict, parser, num_count=parser.sample_image_count_train)

	keys_remove_train = list(set(keys) - set(imgs_best_sample_train.keys()))
	# for k, v in annot_sampled_train.items():
	# 	num = size_dict[k] - v
	# 	if num == 0:
	# 		re = True
	imgs_best_sample_val = sampling(keys_remove_train, dataset_train, size_dict, parser, num_count=parser.sample_image_count_val)


	## Save Train Dataset
	mini_coco = {}
	annots = []
	imgs = []
	# add headers like info, licenses etc.
	mini_coco["info"] = dataset_train.coco.dataset['info']
	mini_coco["licenses"] = dataset_train.coco.dataset['licenses']
	mini_coco["categories"] = dataset_train.coco.dataset['categories']

	for k, v in imgs_best_sample_train.items():
		f_name = dataset_train.coco.imgs[k]['file_name']
		im_id = int(f_name[:-4])
		for ann in dataset_train.coco.imgToAnns[im_id]:
			annots.append(ann)
		imgs.append(dataset_train.coco.imgs[im_id])

	mini_coco['images'] = imgs
	mini_coco['annotations'] = annots

	with open(f"{parser.save_file_name_train}.json", 'w') as jsonf:
		json.dump(mini_coco, jsonf)
  
	
	## Save Validation Dataset
	mini_coco = {}
	annots = []
	imgs = []
	# add headers like info, licenses etc.
	mini_coco["info"] = dataset_train.coco.dataset['info']
	mini_coco["licenses"] = dataset_train.coco.dataset['licenses']
	mini_coco["categories"] = dataset_train.coco.dataset['categories']

	for k, v in imgs_best_sample_val.items():
		f_name = dataset_train.coco.imgs[k]['file_name']
		im_id = int(f_name[:-4])
		for ann in dataset_train.coco.imgToAnns[im_id]:
			annots.append(ann)
		imgs.append(dataset_train.coco.imgs[im_id])

	mini_coco['images'] = imgs
	mini_coco['annotations'] = annots
	with open(f"{parser.save_file_name_val}.json", 'w') as jsonf:
		json.dump(mini_coco, jsonf)
  
  

if __name__ == '__main__':
	main()