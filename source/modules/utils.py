from PIL import Image
import os
import re
import numpy as np
import glob
import pandas as pd

from config import setup_argparse
parser = setup_argparse()
args = parser.parse_args()

def draw_image(x,y,dir,filename):
    path = os.path.join(dir, filename + ".png")
    if args.dataset=='cifar10':
        image = x.reshape([32,32,3])
        im = Image.fromarray((image * 255).astype(np.uint8))
        im.save(path)
    else:
        image = x.reshape([28,28])
        im = Image.fromarray((1-image)*255)
        im.convert('L').save(path)

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_images(path_to_data):
    clean = []
    y = []
    for filename in sorted(glob.glob(os.path.join(path_to_data, '*.png')), key=numericalSort):
        img = Image.open(filename)
        if args.dataset=='cifar10':
            # im_arr = Image.fromarray((image * 255).astype(np.uint8))
            img_arr = np.array(img) / 255.
        else:
            img_arr = 1 - (np.array(img) / 255.)
        clean.append(img_arr.tolist())
        word = re.split('[@.]', filename)
        y_ = int(word[-2])
        y.append(y_)
    x = np.array(clean)
    return x, y


def preprocess_data(model_name, x):
    # reshape input data according to the model's input tensor
    if model_name in ['magnet_1','magnet_2','conditional_cae']:
        x = x.reshape(-1,args.dim12,args.dim12,args.dim3)
    elif model_name == 'autoencoder' or model_name == 'deep_autoencoder':
        x = x.reshape(-1,args.dim12,args.dim12,args.dim3)
    else:
        raise ValueError('Unknown model_name %s was given' % model_name)

    return x


def parse_log_file(log_file_path, col_name):
    log_history = pd.read_csv(log_file_path)
    cols = log_history.columns.values.tolist()
    if not col_name in cols:
        raise ValueError('Unknown metric [{}] from log file [{}] is requested.'.format( col_name, log_file_path) )
    print(cols)
    requested_col = log_history[col_name].values.tolist()
    print(requested_col)
    return requested_col

