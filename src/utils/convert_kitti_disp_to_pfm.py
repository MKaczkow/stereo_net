import os
import re
import sys
import imageio
import numpy as np


KITTI_2012_ROOT = r'D:\engineering_thesis_data\Kitti_2012\data_stereo_flow'
KITTI_2015_ROOT = r'D:\engineering_thesis_data\Kitti_2015\data_scene_flow'


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
        print('greyscale')
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def main():
    
    for root, _, files in os.walk(f'{KITTI_2012_ROOT}/training/disp_noc'):
        for file in files:
            if file.endswith('.png'):
                new_name = file[:-4] + '.pfm'
                depth_png = imageio.v2.imread(f'{root}\{file}')
                depth_png = depth_png.astype(np.float32) / 256.0
                writePFM(f'{root}\{new_name}', depth_png)
                img, scale = readPFM(f'{root}\{new_name}')
                print(img.shape, img.max(), img.min(), img.dtype)


    for root, _, files in os.walk(f'{KITTI_2015_ROOT}/training/disp_noc_0'):
        for file in files:
            if file.endswith('.png'):
                new_name = file[:-4] + '.pfm'
                depth_png = imageio.v2.imread(f'{root}\{file}')
                depth_png = depth_png.astype(np.float32) / 256.0
                writePFM(f'{root}\{new_name}', depth_png)
    
    for root, _, files in os.walk(f'{KITTI_2015_ROOT}/training/disp_noc_1'):
        for file in files:
            if file.endswith('.png'):
                new_name = file[:-4] + '.pfm'
                depth_png = imageio.v2.imread(f'{root}\{file}')
                depth_png = depth_png.astype(np.float32) / 256.0
                writePFM(f'{root}\{new_name}', depth_png)

if __name__ == "__main__":
    main()