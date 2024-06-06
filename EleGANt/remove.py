import os


def load_image_paths(txt_file):
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def save_image_paths(txt_file, paths):
    with open(txt_file, 'w') as f:
        f.write('\n'.join(paths) + '\n')


def check_and_clean_images(image_folder, lms_folder, txt_files):
    # Load image paths from text files
    image_paths = {txt_file: load_image_paths(txt_file) for txt_file in txt_files}

    for txt_file, paths in image_paths.items():
        updated_paths = []
        for img_path in paths:
            npy_path = os.path.join(lms_folder, img_path.replace('.png', '.npy'))
            img_full_path = os.path.join(image_folder, img_path)

            if os.path.exists(npy_path):
                updated_paths.append(img_path)
            else:
                # Remove the .png image if the .npy file does not exist
                if os.path.exists(img_full_path):
                    os.remove(img_full_path)
                    print(f"Deleted image: {img_full_path}")

        # Save the updated paths back to the text file
        save_image_paths(txt_file, updated_paths)
        print(f"Updated {txt_file} with {len(updated_paths)} valid paths.")


if __name__ == "__main__":
    base_path = '/root/cts/ai01/data/MT-Dataset_1024'
    image_folder = os.path.join(base_path, 'images')
    lms_folder = os.path.join(base_path, 'lms')

    txt_files = [
        os.path.join(base_path, 'makeup.txt'),
        os.path.join(base_path, 'non-makeup.txt'),
        os.path.join(base_path, 'train_MAKEMIX.txt'),
        os.path.join(base_path, 'train_SYMIX.txt')
    ]

    check_and_clean_images(image_folder, lms_folder, txt_files)
