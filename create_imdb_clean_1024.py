from typing import List, Union
import cv2
import tqdm
import os
import sys
import os.path as osp
import numpy as np
from PIL import Image, ImageFile, ImageDraw
from multiprocessing import Pool
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMDB_DIR = './data/imdb'

MAX_SIDE = 1024
OUT_DIR = f'./data/imdb-clean-{MAX_SIDE}'
VIS_DIR = f'./data/imdb-clean-{MAX_SIDE}-visualisation'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(osp.join(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def crop_face_with_margin(img, box, crop_margin: Union[float, List[float]] = [0.4, 0.4, 0.4, 0.4]):
    '''
    img: H,W,3 array
    box: x1,y1,x2,y2 list
    adapted from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/extractSubImage.m
    '''
    if isinstance(crop_margin, (float, int)):

        crop_margin = [float(crop_margin)] * 4
    elif isinstance(crop_margin, (list, tuple)):
        assert len(
            crop_margin) == 4, f'crop_margin has to be a float value or a list of four margins, got{crop_margin}'
    else:
        raise ValueError

    h, w = img.shape[:2]

    is_color = len(img.shape) > 2

    # size of face
    orig_size = [0, 0]
    orig_size[0] = box[3]-box[1]+1
    orig_size[1] = box[2]-box[0]+1

    # add margin
    full_crop = [0, 0, 0, 0]
    full_crop[0] = round(box[0]-crop_margin[0]*orig_size[1])
    full_crop[1] = round(box[1]-crop_margin[1]*orig_size[0])
    full_crop[2] = round(box[2]+crop_margin[2]*orig_size[1])
    full_crop[3] = round(box[3]+crop_margin[3]*orig_size[0])

    # size of face with margin
    new_size = [0, 0]
    new_size[0] = full_crop[3]-full_crop[1]+1
    new_size[1] = full_crop[2]-full_crop[0]+1

    # ensure that the region cropped from the original image with margin doesn't go beyond the image size
    crop = [0, 0, 0, 0]
    crop[0] = max(full_crop[0], 0)
    crop[1] = max(full_crop[1], 0)
    crop[2] = min(full_crop[2], w-1)
    crop[3] = min(full_crop[3], h-1)

    # size of the actual region being cropped from the original image
    crop_size = [0, 0]
    crop_size[0] = crop[3]-crop[1]+1
    crop_size[1] = crop[2]-crop[0]+1

    if is_color:
        new_img = np.zeros(new_size+[3], dtype=np.uint8)
    else:
        new_img = np.zeros(new_size, dtype=np.uint8)
    # coordinates of region taken out of the original image in the new image
    new_location = [0, 0, 0, 0]
    new_location[0] = crop[0]-full_crop[0]
    new_location[1] = crop[1]-full_crop[1]
    new_location[2] = crop[0]-full_crop[0]+crop_size[1]-1
    new_location[3] = crop[1]-full_crop[1]+crop_size[0]-1

    # # coordinates of the face in the new image
    new_box = [0, 0, 0, 0]
    new_box[0] = new_location[0]+box[0]-crop[0]
    new_box[1] = new_location[1]+box[1]-crop[1]
    new_box[2] = new_location[2]+box[2]-crop[2]
    new_box[3] = new_location[3]+box[3]-crop[3]
    new_box = np.array(new_box, int)
    # do the crop
    new_img[new_location[1]:new_location[3], new_location[0]:new_location[2], ...] = \
        img[crop[1]:crop[3], crop[0]:crop[2], ...]
    return new_img, new_box


def resize_max_side(im: np.array, max_side: int,
                    bbox: Union[np.array, list] = None,
                    square: bool = False,
                    inter=cv2.INTER_LINEAR,
                    pad_mode: str = 'constant'):
    h, w = im.shape[:2]
    if h > w:
        scale = max_side / float(h)
        sz = (int(w * scale), max_side)
    else:
        scale = max_side / float(w)
        sz = (max_side, int(h * scale))

    im = cv2.resize(im, sz, inter)

    if bbox is not None:
        x1, y1, x2, y2 = bbox[:4]
        x1, x2 = x1 * scale, x2 * scale
        y1, y2 = y1 * scale, y2 * scale
        bbox = np.array([x1, y1, x2, y2], int)

    if square:
        w_diff, h_diff = max_side-sz[0], max_side-sz[1]
        h_pad = (h_diff//2, h_diff-h_diff//2)
        w_pad = (w_diff//2, w_diff-w_diff//2)
        pad_sz = [h_pad, w_pad]
        if len(im.shape) > 2:
            pad_sz += [(0, 0)]
        im = np.pad(im, pad_sz, pad_mode)
        if bbox is not None:
            bbox[0] += w_pad[0]
            bbox[2] += w_pad[0]
            bbox[1] += h_pad[0]
            bbox[3] += h_pad[0]
    return im, bbox


def process(ind):
    row = imdb_csv_split.iloc[ind].copy()
    fn = osp.join(IMDB_DIR, row.filename)
    img = np.array(pil_loader(fn))
    x_min, y_min, x_max, y_max = map(
        int, [row.x_min, row.y_min, row.x_max, row.y_max])
    bbox = [x_min, y_min, x_max, y_max]
    img, bbox = crop_face_with_margin(img, bbox, crop_margin=1.)
    h, w = img.shape[:2]
    if h > MAX_SIDE or w > MAX_SIDE:
        img, bbox = resize_max_side(img, MAX_SIDE, bbox, square=True)
    row.x_min, row.y_min, row.x_max, row.y_max = bbox
    out_fn = fn.replace(IMDB_DIR, OUT_DIR)

    os.makedirs(osp.dirname(out_fn), exist_ok=1)
    img = Image.fromarray(img)

    img.save(out_fn)

    # visualisating some random images 
    if ind % 100 == 0:
        draw = ImageDraw.Draw(img)
        draw.text((row.x_min, row.y_min), f'Age: {row.age}')
        draw.rectangle((row.x_min, row.y_min, row.x_max, row.y_max), outline=(255, 255, 255))
        vis_fn = out_fn.replace(OUT_DIR, VIS_DIR)
        os.makedirs(osp.dirname(vis_fn), exist_ok=1)
        img.save(vis_fn)

    return row


if __name__ == '__main__':
    output = []
    split = sys.argv[1]
    imdb_csv_split = pd.read_csv(f'./csvs/imdb_{split}_new.csv')

    with Pool(processes=16) as pool:
        for row in tqdm.tqdm(pool.imap_unordered(process, imdb_csv_split.index), total=len(imdb_csv_split.index)):
            output.append(row)
    output = pd.concat(output, axis=1).transpose()
    output.to_csv(f'{OUT_DIR}/imdb_{split}_new_{MAX_SIDE}.csv', index=False)
