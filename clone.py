import os
import random
import shutil

def check_sync_and_fix(image_dir, lms_dir):
    missing_files = []

    for category in ['makeup', 'non-makeup']:
        image_path = os.path.join(image_dir, category)
        lms_path = os.path.join(lms_dir, category)

        # Lấy danh sách các tệp ảnh và .npy trong từng thư mục
        image_files = {os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.png')}
        npy_files = {os.path.splitext(f)[0] for f in os.listdir(lms_path) if f.endswith('.npy')}

        # Tìm các tệp ảnh không có tệp .npy tương ứng
        missing_files.extend(image_files - npy_files)

        # Xử lý các tệp bị thiếu
        if missing_files:
            print(f"Processing missing .npy files in category '{category}':")
            for missing_file in missing_files:
                # Chọn một file .npy ngẫu nhiên để clone
                if npy_files:
                    random_npy_file = random.choice(list(npy_files))
                    src_file = os.path.join(lms_path, f"{random_npy_file}.npy")
                    dst_file = os.path.join(lms_path, f"{missing_file}.npy")
                    shutil.copy(src_file, dst_file)
                    print(f"Cloned {src_file} to {dst_file}")
                else:
                    print(f"No existing .npy files to clone from in category '{category}'.")

    return missing_files

if __name__ == "__main__":
    root_path = '/root/cts/ai01/data/MT-Dataset_1024'
    images_dir = os.path.join(root_path, 'images')
    lms_dir = os.path.join(root_path, 'lms')

    missing_files = check_sync_and_fix(images_dir, lms_dir)

    if missing_files:
        print("Missing .npy files have been cloned and replaced.")
    else:
        print("All images have corresponding .npy files.")
