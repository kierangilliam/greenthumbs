"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                      Imports
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import detectron2
# detectron2 utilities
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# common libs
import numpy as np
import cv2
import random
import os
import json


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                      Constants 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
BASE_DIR = 'greenthumbs/data'
NUM_CLASSES = 5


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                               Hyper parameter selection
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Versions contain different amounts of data
# or different data augmentation methods
versions = ["v01"]
test_train_splits = ["65%", "75%", "85%"]
iters = [750, 1250]
lrs = [.002, .001, .0005, .00025, .0001]
batch_sizes_per_img = [128, 512]
models = [ # All presumed to be in the COCO_Detection folder
  "faster_rcnn_R_101_FPN_3x.yaml",   
  "faster_rcnn_R_50_C4_3x.yaml",
  "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
  # "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
]

## TODO: RetinaNet by its definition does not have ROI_HEADS. 
# Need to use cfg.MODEL.RETINANET.SCORE_THRESH_TEST



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                               Lib
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
ds: 65%, 75%, 85%... Describes the dataset split
"""
def get_train_test_coco_files(v, ds):
  train_coco_file = BASE_DIR + f'/{v}/train_{ds}_coco.json'
  test_coco_file  = BASE_DIR + f'/{v}/test_{ds}_coco.json'
  return train_coco_file, test_coco_file


def save_result(instance, result):
  print('Saving results...')
  filename = f'{instance}.json'
  contents = json.dumps(result)

  with open(filename, 'w') as f:
    f.write(contents)

  print(f'Saved {filename}')


def train(instance, v, ds, model, iterations, lr, batch_size, dry_run=False):
  img_dir  = BASE_DIR + f'/{v}/ds/'
  model_path = f'COCO-Detection/{model}'

  train_instance = f'train/{instance}'
  test_instance  = f'test/{instance}'

  train_coco_file, test_coco_file = get_train_test_coco_files(v, ds)

  register_coco_instances(train_instance, {}, train_coco_file, img_dir)
  register_coco_instances(test_instance, {}, test_coco_file, img_dir)

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(model_path))
  cfg.DATASETS.TRAIN = (train_instance,)
  cfg.DATASETS.TEST = ()
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.OUTPUT_DIR = f'./outputs/{instance}'

  # Hyperparameter seleciton  
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path) # TODO What is checkpoint url?
  cfg.SOLVER.MAX_ITER = iterations
  cfg.SOLVER.BASE_LR = lr
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = DefaultTrainer(cfg) 
  trainer.resume_or_load(resume=False)

  if not dry_run:
    print(f'**********************************************')
    print(f'\t\t Begin train {instance}'                 )
    print(f'**********************************************')
    trainer.train()

  return cfg, trainer, train_instance, test_instance



def test(instance, cfg, trainer, test_instance):  
  print(f'**********************************************')
  print(f'\t\t Test {instance}'                          )
  print(f'**********************************************')
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75 # threshold of 75% certainty at which we’ll consider the predictions as correct
  cfg.DATASETS.TEST = (test_instance, )
  # predictor = DefaultPredictor(cfg)
  # predictor = ExtendedPredictor(cfg)
  evaluator = COCOEvaluator(test_instance, cfg, False, output_dir=cfg.OUTPUT_DIR)
  test_loader = build_detection_test_loader(cfg, test_instance)
  result = inference_on_dataset(trainer.model, test_loader, evaluator)  
  
  return result


for v in versions:
  for ds in test_train_splits:
    for model in models:
      for iterations in iters:
        for lr in lrs:
          for batch_size in batch_sizes_per_img:
            
            instance = f'{v}_{ds}_{model}_{iterations}_{lr}_{batch_size}'

            cfg, trainer, train_instance, test_instance = train(
                instance, v, ds, model, iterations, lr, batch_size
            )

            result = test(instance, cfg, trainer, test_instance)

            save_result(instance, result)

                       
