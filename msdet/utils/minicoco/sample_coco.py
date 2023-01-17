import json
import argparse
from random import shuffle

from dataloader import CocoDataset
from sampler_utils import get_coco_object_size_info, get_coco_class_object_counts

# default area ranges defined in coco
areaRng = [32 ** 2, 96 ** 2, 1e5 ** 2]


def main(args=None):
	parser = argparse.ArgumentParser(description='Mini COCO Sampling Script')

	parser.add_argument('--coco_path', help='Path to COCO directory', default="/default/path/to/COCO2017/")
	parser.add_argument('--save_file_name', help='Save file name', default="instances_train2017_minicoco")
	parser.add_argument('--save_format', help='Save to json or csv', default="json")
	parser.add_argument('--sample_image_count', help='How many images you want to sample', type=int, default=25000)
	parser.add_argument('--run_count', help='How many times you want to run sampling', type=int, default=10000000)
	parser.add_argument('--debug', help='Print useful info', action='store_true')

	parser = parser.parse_args(args)

	dataset_train = CocoDataset(parser.coco_path, set_name='train2017')

	# get coco class based object counts
	annot_dict = get_coco_class_object_counts(dataset_train)
	if parser.debug:
		print(f"COCO object counts in each class:\n{annot_dict}")

	# fig = plt.figure()
	# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
	# axes.plot(np.arange(0,80,1), np.divide(annot_dict.values(), float(len(dataset_train.image_ids)) ), 'r')
	# axes.plot(np.arange(0,80,1), np.divide(annot_dict.values(), float(len(dataset_train.image_ids)) ), 'r')
	# axes.set_xlabel('Class Id')
	# axes.set_ylabel('Annot Count')
	# axes.set_title('MiniCoco vs Coco 2017 train set')
	# fig.show()
	# fig.savefig("minicoco_vs_coco_train2017_annot.png")

	# here extract object sizes.
	size_dict = get_coco_object_size_info(dataset_train)
	if parser.debug:
		print(f"COCO object counts in each class for different sizes (S,M,L):\n{size_dict}")

	# now sample!!
	imgs_best_sample = {}
	ratio_list = []
	best_diff = 1_000_000
	keys = []
	# get all keys in coco train set, total image count!
	for k, v in dataset_train.coco.imgToAnns.items():
		keys.append(k)

	for rr in range(parser.run_count):
		imgs = {}

		# shuffle keys
		shuffle(keys)

		# select first N images
		for i in keys[:parser.sample_image_count]:
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
		if parser.debug:
			print(f"Sampled Annotations dict:\n {annot_sampled}")

		# calculate ratios
		ratios_obj_count = {}
		# ratios_obj_size = {}

		failed_run = False
		for k, v in size_dict.items():
			if not k in annot_sampled:
				failed_run = True
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

		if parser.debug:
			print(f"Best difference:{best_diff}")

	if parser.save_format == 'csv':
		# now write to csv file
		csv_file = open(f"{parser.save_file_name}.csv", 'w')
		write_str = ""

		for k, v in imgs_best_sample.items():
			f_name = dataset_train.coco.imgs[k]['file_name']
			for ann in v:
				bbox = ann['bbox']
				class_id = ann['category_id']
				write_str = f_name + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(
					bbox[3]) + ',' + \
							str(dataset_train.labels[dataset_train.coco_labels_inverse[class_id]]) + ',' + '0' + '\n'

				csv_file.write(write_str)

		csv_file.close()

	elif parser.save_format == 'json':
		mini_coco = {}
		annots = []
		imgs = []
		# add headers like info, licenses etc.
		mini_coco["info"] = dataset_train.coco.dataset['info']
		mini_coco["licenses"] = dataset_train.coco.dataset['licenses']
		mini_coco["categories"] = dataset_train.coco.dataset['categories']

		for k, v in imgs_best_sample.items():
			f_name = dataset_train.coco.imgs[k]['file_name']
			im_id = int(f_name[:-4])
			for ann in dataset_train.coco.imgToAnns[im_id]:
				annots.append(ann)
			imgs.append(dataset_train.coco.imgs[im_id])

		mini_coco['images'] = imgs
		mini_coco['annotations'] = annots

		with open(f"{parser.save_file_name}.json", 'w') as jsonf:
			json.dump(mini_coco, jsonf)


if __name__ == '__main__':
	main()