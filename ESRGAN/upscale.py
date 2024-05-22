import os
import os.path as osp
import glob
from concurrent.futures import ThreadPoolExecutor
import time
import math

import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

ORIGINAL_PATH = '/root/cts/ai01/data'

IMG_SIZE = 640

def process_images_in_batch(paths, device, model, start_idx):
    for idx, path in enumerate(paths, start=start_idx):
        base = osp.splitext(osp.basename(path))[0]
        print(f'Processing {base} - {idx}/{start_idx + len(paths) - 1}')

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        height, width, channels = img.shape
        if width == IMG_SIZE and height == IMG_SIZE:
            print('Converted!')
            continue

        output_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, output_resized)
        continue

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        output_resized = cv2.resize(output, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, output_resized)

def process_folder_batch(folder_path, device, model, batch_size):
    paths = glob.glob(folder_path + '/*')
    total = len(paths)
    num_batches = math.ceil(total / batch_size)

    with ThreadPoolExecutor(max_workers=num_batches) as executor:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total)
            batch_paths = paths[start_idx:end_idx]

            executor.submit(process_images_in_batch, batch_paths, device, model, start_idx + 1)

torch.cuda.empty_cache()

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

makeup = ORIGINAL_PATH + '/MT-Dataset/images/makeup'
non_makeup = ORIGINAL_PATH + '/MT-Dataset/images/non-makeup'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nBegin upscale...'.format(model_path))
process_folder_batch(makeup, device, model, 10)
process_folder_batch(non_makeup, device, model, 10)
