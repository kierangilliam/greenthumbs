'''
within the data/v*/ folder...
$ python3 ../../scripts/merge_datasets.py
'''

import shutil
import os
import json
from os import listdir
from os.path import isfile, join

tomato_folders = [
    'main_vid',
    '0420-0566',
    'ds1',
    'labeled ds2',
    'classified',
    'labeled mainly seedstart',
]
bell_folders = ['bell_peppers_2', 'bell_peppers', 'bell_peppers_001']
cucumber_folders = ['Cucumbers1', 'Cucumbers2', 'Cucumbers3']

folders = []
folders.extend(tomato_folders)
folders.extend(bell_folders)
folders.extend(cucumber_folders)

print(folders)

dst = os.path.join(os.getcwd(), 'ds')

try:
    shutil.rmtree(dst)
except Exception as e:
    print("No folder to delete")

os.mkdir(dst)

total_files = 0
processed_files = 0

for dataset in folders:
    path = os.path.join(os.getcwd(), dataset)
    ann_path = os.path.join(path, './ann')
    files = [f for f in listdir(ann_path) if isfile(join(ann_path, f))]
    processed = 0

    print(dataset)
    print(ann_path)
    print(len(files))

	# mv path -> dst
	# rename new files
    for i, file in enumerate(files):

        src_file_path = ann_path + "/" + file

        with open(src_file_path, 'r') as f:
            data = json.loads(f.read())

        if len(data['objects']) > 0:
            src_img_file = file[:-5]
            dst_img_file = f"{i + processed_files}__{src_img_file}"
            filename = f"{i + processed_files}__{file}"
            shutil.copy2(src_file_path, dst + "/" + filename)
            shutil.copy2(path + "/img/" + src_img_file, dst + "/" + dst_img_file)
            processed += + 1
            print(f'\tProcessed {filename}...')

    total_files += len(files)
    processed_files += processed

print(f'{total_files} total files... {processed_files} had objects')
