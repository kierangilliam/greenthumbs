import random
import json

# Creates two different coco files out of a single coco file
base_dir = "data/v02/"

ratios = [.65, .75, .85]

coco_file = base_dir + '/coco.json'

with open(coco_file) as f:
  dataset = json.load(f)

for ratio in ratios:
    train, test = {}, {}
    train["categories"] = dataset["categories"]
    test["categories"] = dataset["categories"]

    train_len = int(len(dataset["images"]) * ratio)
    random.shuffle(dataset["images"])

    train["images"] = dataset["images"][:train_len]
    test["images"] = dataset["images"][train_len:]

    for ds in [train, test]:
      annotations = []
      for image in ds["images"]:
        i = image["id"]
        annotations.extend([ann for ann in dataset["annotations"] if ann["image_id"] == i])
      ds["annotations"] = annotations

    with open(base_dir + f'/train_{int(ratio * 100)}%_coco.json', 'w') as outfile:
      json.dump(train, outfile)

    with open(base_dir + f'/test_{int(ratio * 100)}%_coco.json', 'w') as outfile:
      json.dump(test, outfile)

    print(f"Done. {len(train['images'])} train images, {len(test['images'])} test images.")
