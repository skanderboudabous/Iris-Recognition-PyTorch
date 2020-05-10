#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
from tqdm import tqdm
from glob import glob
import os, cv2, random
from collections import defaultdict
random.seed(0)


#------------------------------------------------------------------------------
#  Parameters
#------------------------------------------------------------------------------
IMAGE_DIR = "/media/antiaegis/storing/datasets/Iris/CASIA4/Interval"


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
	# Get files
	files = sorted(glob(os.path.join(IMAGE_DIR, "**/*.*"), recursive=True))
	files = [file for file in files if 'jpg' in file]
	print("Number of files:", len(files))

	# Aggregate data
	data_dict = defaultdict(lambda: defaultdict(lambda: list()))
	for file in files:
		basename = os.path.basename(file)[:-4]
		person_id = int(basename[2:5])
		eye_id = 0 if basename[5]=='L' else 1
		ins_id = int(basename[-2:])
		data_dict[person_id][eye_id].append(file)

	for vals in data_dict.values():
		for val in vals.values():
			random.shuffle(val)

	# Split train/valid
	train_list = []
	valid_list = []

	for key, vals in data_dict.items():
		for val in vals.values():
			train_samples = int(len(val) * 0.6)
			train_list += [val[i] for i in range(train_samples)]
			valid_list += [val[i] for i in range(train_samples, len(val))]

	# Write to file
	random.shuffle(train_list)
	with open("data/casia4_interval_train.txt", 'w') as fp:
		for file in train_list:
			fp.writelines(file+'\n')

	random.shuffle(valid_list)
	with open("data/casia4_interval_valid.txt", 'w') as fp:
		for file in valid_list:
			fp.writelines(file+'\n')
