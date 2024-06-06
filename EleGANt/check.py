import os

def check_sync(image_dir, lms_dir):
    missing_files = []

    for category in ['makeup', 'non-makeup']:
        image_path = os.path.join(image_dir, category)
        lms_path = os.path.join(lms_dir, category)

        # Lấy danh sách các tệp ảnh và .npy trong từng thư mục
        image_files = {os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.png')}
        npy_files = {os.path.splitext(f)[0] for f in os.listdir(lms_path) if f.endswith('.npy')}

        # Tìm các tệp ảnh không có tệp .npy tương ứng
        missing_files.extend(image_files - npy_files)

    return missing_files

if __name__ == "__main__":
    root_path = '/root/cts/ai01/data/MT-Dataset_1024'
    images_dir = os.path.join(root_path, 'images')
    lms_dir = os.path.join(root_path, 'lms')

    missing_files = check_sync(images_dir, lms_dir)

    if missing_files:
        print("Missing .npy files for the following images:")
        for file in missing_files:
            print(file)
    else:
        print("All images have corresponding .npy files.")