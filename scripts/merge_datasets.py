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
	for subfolder in ['ann', 'img']:

		print(dataset)
		path = os.path.join(os.getcwd(), dataset + '/' + subfolder) 
		files = [f for f in listdir(path) if isfile(join(path, f))]
		print(len(files))
	    
		# mv path -> dst
		# rename new files
		for i, file in enumerate(files):
			filename = f"{i + total_files}__{file}"
			print(filename)
			shutil.copy2(path + "/" + file, dst + "/" + filename)
 
	total_files += len(files)
