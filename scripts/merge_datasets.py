import shutil 
import os
import json
from os import listdir
from os.path import isfile, join

folders = ['ds1', 'ds2']

dst = os.path.join(os.getcwd(), 'ds')

try:
    shutil.rmtree(dst)
except Exception as e:
    print("No folder to delete")

os.mkdir(dst)

total_files = 0

for dataset in folders:

	print(dataset)
	path = os.path.join(os.getcwd(), dataset) 
	ann_path = os.path.join(path, './ann')
	print(ann_path)
	files = [f for f in listdir(ann_path) if isfile(join(ann_path, f))]
	print(len(files))
    
	# mv path -> dst
	# rename new files
	for i, file in enumerate(files):
		src_img_file = file[:-5] 
		dst_img_file = f"{i + total_files}__{src_img_file}"	
		filename = f"{i + total_files}__{file}"
		print(filename)
		shutil.copy2(ann_path + "/" + file, dst + "/" + filename)
		shutil.copy2(path + "/img/" + src_img_file, dst + "/" + dst_img_file)
 
	total_files += len(files)
