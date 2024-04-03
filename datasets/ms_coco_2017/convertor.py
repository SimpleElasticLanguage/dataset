#!/usr/bin/python3

import json
from collections import defaultdict

OUTPUTPATH = "ms_coco_2017.ndjson"

# For all files
# .images
# {
#   "license": 3,
#   "file_name": "000000391895.jpg",
#   "coco_url": "http://images.cocodataset.org/train2017/000000391895.jpg",
#   "height": 360,
#   "width": 640,
#   "date_captured": "2013-11-14 11:18:45",
#   "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
#   "id": 391895
# }

# captions
# .annotations
# {
#   "image_id": 16977,
#   "id": 89,
#   "caption": "A car that seems to be parked illegally behind a legally parked car"
# }

# .categories
# {
#   "supercategory": "person",
#   "id": 1,
#   "name": "person"
# }

# instances
# .annotations
# {
#   "segmentation": [
#     [
#       239.97,
#       260.24,
#       222.04,
#       270.49,
#       199.84,
#       253.41,
#       213.5,
#       227.79,
#       259.62,
#       200.46,
#       274.13,
#       202.17,
#       277.55,
#       210.71,
#       249.37,
#       253.41,
#       237.41,
#       264.51,
#       242.54,
#       261.95,
#       228.87,
#       271.34
#     ]
#   ],
#   "area": 2765.1486500000005,
#   "iscrowd": 0,
#   "image_id": 558840,
#   "bbox": [
#     199.84,
#     200.46,
#     77.71,
#     70.88
#   ],
#   "category_id": 58,
#   "id": 156
# }


def main():
    train_images, train_instances = read_instances("instances_train2017.json", "train2017")
    val_images, val_instances = read_instances("instances_val2017.json", "val2017")
    images = {**train_images, **val_images}
    instances = {**train_instances, **val_instances}

    train_images, train_captions = read_captions("captions_train2017.json", "train2017")
    val_images, val_captions = read_captions("captions_val2017.json", "val2017")
    images = {**images, **train_images, **val_images}
    captions = {**train_captions, **val_captions}

    images = merge(images, captions, instances)

    with open(OUTPUTPATH, "w") as fd:
        for _, image in images.items():
            fd.write(json.dumps(image) + "\n")


def read_instances(filepath, directory_name):

    with open(filepath) as fd:
        data = json.load(fd)
        images = get_images(data, directory_name)
        instances = get_instances(data)

        return images, instances


def read_captions(filepath, directory_name):

    with open(filepath) as fd:
        data = json.load(fd)
        images = get_images(data, directory_name)
        captions = get_captions(data)

        return images, captions


def merge(images, captions, instances):

    for iid, caps in captions.items():
        images[iid]["captions"] = caps

    for iid, cats in instances.items():
        images[iid]["categories"] = cats

    return images


def get_captions(data):
    captions = defaultdict(list)

    for anno in data["annotations"]:
        captions[anno["image_id"]].append(anno["caption"])

    return captions


def get_instances(data):
    categories = {}
    annotations = defaultdict(list)

    for cat in data["categories"]:
        categories[cat["id"]] = {"name": cat["name"], "supercategory": cat["supercategory"]}

    for anno in data["annotations"]:
        obj = {k: v for k, v in anno.items() if "id" not in k}
        category = categories[anno["category_id"]]
        obj["category"] = category["name"]
        obj["supercategory"] = category["supercategory"]
        obj["iscrowd"] = obj["iscrowd"] == 1
        annotations[anno["image_id"]].append(obj)

    for iid, categories in annotations.items():
        category_counts = defaultdict(int)
        supercategory_counts = defaultdict(int)

        for cat in categories:
            category_counts[cat["category"]] += 1
            supercategory_counts[cat["supercategory"]] += 1

        for cat in categories:
            cat["count"] = category_counts[cat["category"]]
            cat["supercount"] = supercategory_counts[cat["supercategory"]]

    return annotations


def get_images(data, directory_name):
    licenses = {l["id"]: {"url": l["url"], "name": l["name"]} for l in data["licenses"]}

    images = {}
    for img in data["images"]:
        image = image_formator(img, licenses, directory_name)
        images[image["id"]] = image

    return images


def image_formator(image, licenses, directory_name):
    return {
        "url": image["coco_url"],
        "filepath": f'{directory_name}/{image["file_name"]}',
        "width": image["width"],
        "height": image["height"],
        "license": licenses[image["license"]],
        "date_captured": image["date_captured"],
        "id": image["id"],
    }


if __name__ == "__main__":
    main()
