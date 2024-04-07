#!/usr/bin/python3

import os
import json
import numpy as np
from collections import defaultdict, Counter
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage import color
import webcolors


INPUT_FILE = "ms_coco_2017.ndjson"
OUTPUT_FILE = "ms_coco_2017_colorized.ndjson"
PANTONE_FILE = "pantone_colors.json"


def main():
    palettes = {
        "pantone": init_palette(read_json(PANTONE_FILE)),
        "css3": init_palette(webcolors.CSS3_HEX_TO_NAMES),
        "css2": init_palette(webcolors.CSS2_HEX_TO_NAMES),
    }

    dataset = read_ndjson(INPUT_FILE)

    if os.path.isfile(OUTPUT_FILE):
        print("Filtering out existing data")
        existing_colorized_dataset = read_ndjson(OUTPUT_FILE)
        dataset = filter_out_by(dataset, existing_colorized_dataset, lambda d: d["filepath"])

    print("Predicting ...")
    colorized_dataset = predict_colors(dataset, palettes)
    write_ndjson(OUTPUT_FILE, colorized_dataset, mode="a")
    print("Done")


def init_palette(palette):
    colors = list(palette.keys()) # Get palette hexa color

    return {
        "palette": palette, # key = hexa, value = color name
        "colors": colors,
        "lab": hexa_palette_to_lab(colors)
    }


def filter_out_by(dataset1, dataset2, get_key):
    keys = [get_key(d) for d in dataset2]

    for line in dataset1:
        if get_key(line) not in keys:
            yield line


def predict_colors(data, palettes):

    for line in data:

        counts = defaultdict(int)
        img = Image.open("../" + line["filepath"])
        name = os.path.basename(line["filepath"])
        print("\n" + name)

        # directory = f"tmp/{name}"
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # img.save(f"{directory}/{name}")

        for cat in line.get("categories", []):

            if "segmentation" in cat:

                counts[cat["category"]] += 1
                catimg = mask_and_crop(img, cat["segmentation"])
                # catimg.save(f"{directory}/{cat['category']}-{counts[cat['category']]}.png")

                try:

                    main_color, kimg = kmeans(catimg, palettes)
                    # kimg.save(f"{directory}/{cat['category']}-{counts[cat['category']]}-kimg.png")
                    print(f'{cat["category"]}-{counts[cat["category"]]} {main_color}')

                    cat["color"] = {
                        "pantone": main_color[0],
                        "css3": main_color[1],
                        "css2": main_color[2],
                        "hexa": main_color[3]
                    }

                except (ValueError, IndexError) as exc:
                    pass

        yield line
        # break


def kmeans(img, palettes):
    # Use all CPU
    kmeans = MiniBatchKMeans(n_clusters=4, n_init=2, random_state=42)

    # Reshape to 2D np array + filter out transparent pixels
    data = img.getdata()
    img2D = np.asarray([(r,g,b) for r,g,b,a in data if a != 0], np.uint8)

    # Operate K-means
    labels = kmeans.fit_predict(img2D)
    rgb_cols = kmeans.cluster_centers_.round(0).astype(int)

    # Get bigest centroid
    counter = Counter(labels)
    idx = max(counter, key=counter.get)

    # Get color name in 3 different numenclatures
    rgb = rgb_cols[idx]
    main_color = (
        to_color_name(rgb, palettes["pantone"]),
        to_color_name(rgb, palettes["css3"]),
        to_color_name(rgb, palettes["css2"]),
        f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    )

    return main_color, None

    # Generate output image with transparent pixels
    # height, width = img.size
    # componed = np.array(list(fill_transparent_pixel(data, rgb_cols[labels])), np.uint8)
    # componed = np.reshape(componed, (width, height, 4)) # + rgba dimensions
    # new_img = Image.fromarray(componed.astype('uint8'), 'RGBA')

    # return main_color, new_img


def fill_transparent_pixel(img_source, img_quant):
    y = 0

    for _, _, _, a in img_source:

        if a == 0:
            yield (0, 0, 0, 0) # Black alpha 0

        else:
            r, g, b, = img_quant[y]
            yield (r, g, b, 255) # Real pixel + alpha 255
            y += 1


def mask_and_crop(input_img, segmentations):
    img = input_img.copy()

    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)

    all_segmentations = []
    for seg in segmentations:
        all_segmentations += seg
        draw.polygon(seg, fill=255)

    img.putalpha(mask)

    return img.crop(bounding_box(all_segmentations))


def bounding_box(seg):
    all_x = [x for i, x in enumerate(seg) if i % 2 == 0]
    all_y = [y for i, y in enumerate(seg) if i % 2 == 1]

    return min(all_x), min(all_y), max(all_x), max(all_y)


def read_ndjson(filepath):
    with open(filepath) as fin:
        for line in fin:
            yield json.loads(line)


def write_ndjson(filepath, data, mode="w"):
    with open(filepath, mode) as fout:
        for line in data:
            fout.write(json.dumps(line) + "\n")


def read_json(filepath):
    with open(filepath) as fin:
        return json.load(fin)


def hexa_palette_to_lab(hex_rgb_colors):
    r = np.asarray([int(h[1:3], 16) for h in hex_rgb_colors], np.uint8)  # List of red elements.
    g = np.asarray([int(h[3:5], 16) for h in hex_rgb_colors], np.uint8)  # List of green elements.
    b = np.asarray([int(h[5:7], 16) for h in hex_rgb_colors], np.uint8)  # List of blue elements.

    rgb = np.dstack((r, g, b)) # Create 3D array

    return color.rgb2lab(rgb)


def to_color_name(rgb, palette):
    p_lab = palette["lab"]

    rgb = np.asarray(rgb, np.uint8)
    rgb = np.dstack((rgb[0], rgb[1], rgb[2])) # Create 3D array
    lab = color.rgb2lab(rgb)

    # Compute Euclidean distance from lab to each element of p_lab
    lab_dist = (
        (p_lab[:,:,0] - lab[:,:,0]) ** 2 +
        (p_lab[:,:,1] - lab[:,:,1]) ** 2 +
        (p_lab[:,:,2] - lab[:,:,2]) ** 2
    ) ** 0.5

    # Get the index of the minimum distance
    min_index = lab_dist.argmin()

    # Get hexa of the color with the minimum distance
    closest_hex = palette["colors"][min_index]

    # Get color name of the minimum distance
    color_name = palette["palette"][closest_hex].lower()

    return color_name


if __name__ == "__main__":
    main()
