import os, sys, datetime
import argparse
from PIL import Image
from shutil import copyfile, disk_usage


def create_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


def print_os(info):
    os.system(f"echo {str(info)}")


def run_command(command: str):
    print_os(f"[DEBUG] running command: {command}")
    ret = os.system(command)
    return ret


def get_dataset_dst_folder():
    _, _, free = disk_usage('/dev/temp_data')
    if free >= 384 * 1000 * 1000 * 1000:
        return '/dev/temp_data/'
    _, _, free = disk_usage('/dev/shm')
    if free >= 384 * 1000 * 1000 * 1000:
        return '/dev/shm/'
    assert False


def install_dependency():
    run_command('pwd')
    run_command('ls')
    os.system('python -m pip install -U pip')
    # os.system('python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116')
    run_command('sudo /opt/conda/bin/pip uninstall torchtext -y')
    run_command('sudo /opt/conda/bin/pip install -r requirements.txt')
    run_command('nvidia-smi')


def find_ckpt(ckpt_path):
    if os.path.isfile(ckpt_path):
        print_os(f"[DEBUG] ckpt file exist {ckpt_path}")
        return ckpt_path

    print_os(f"[DEBUG] ckpt file NOT exist! Now try to search...")
    base_dir, _ = os.path.split(ckpt_path)
    f_list = os.listdir(base_dir)
    for f in f_list:
        if f.endswith(".ckpt"):
            guess_ckpt = os.path.join(base_dir, f)
            if os.path.isfile(guess_ckpt):
                print_os(f"[DEBUG] find ckpt file: {guess_ckpt}")
                return guess_ckpt
            else:
                continue
    
    print_os(f"[ERROR] NO ckpt find!")
    return None


def find_images(img_folder, output_file):
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    counter = 0
    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(img_folder):
            for name in files:
                if name.lower().endswith(image_extensions):
                    image_path = os.path.abspath(os.path.join(root, name))
                    try:
                        Image.open(image_path).verify()
                        file.write(image_path + '\n')
                        counter += 1
                    except:
                        print_os(f"[DEBUG] find corrupted image {image_path}! skip this image.")
    print_os(f"[DEBUG] find {counter} images in {img_folder}, writing their path to {output_file}")
    return output_file
