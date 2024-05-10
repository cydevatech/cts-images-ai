import os.path as osp
import glob
import threading
import time

import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

ORIGINAL_PATH = '/content/drive/My Drive/AI/data'


def process_folder(folder_path):
    total = len(glob.glob(folder_path + '/*'))
    idx = 0
    start_time = time.time()


    for path in glob.glob(folder_path+ '/*'):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(f'Processing {base} - {idx}/{total}')
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        output_resized = cv2.resize(output, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, output_resized)

        print(path)

        elapsed_time = time.time() - start_time
        time_per_image = elapsed_time / idx
        remaining_images = total - idx
        remaining_time = remaining_images * time_per_image

        # Chuyển đổi thời gian còn lại thành giờ và phút
        hours, rem = divmod(remaining_time, 3600)
        minutes, _ = divmod(rem, 60)

        print(f'Remaining time: {int(hours)} hours {int(minutes)} minutes')


def process_folders_concurrently():
    makeup_thread = threading.Thread(target=process_folder, args=(makeup,))
    non_makeup_thread = threading.Thread(target=process_folder, args=(non_makeup,))

    makeup_thread.start()
    non_makeup_thread.start()

    makeup_thread.join()
    non_makeup_thread.join()


model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

makeup = ORIGINAL_PATH + '/MT-Dataset/images/makeup'
non_makeup = ORIGINAL_PATH + '/MT-Dataset/images/non-makeup'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nBegin upscale...'.format(model_path))
process_folders_concurrently()
