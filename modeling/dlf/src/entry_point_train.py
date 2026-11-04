import os, sys, datetime
import argparse
from shutil import copyfile, disk_usage

from entry_point_helper import print_os, run_command, get_dataset_dst_folder, install_dependency, find_images

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--train_dataset_dir", type=str)
    parser.add_argument("--train_dataset_extra_dir", type=str, default="")
    parser.add_argument("--val_dataset_dir", type=str)
    parser.add_argument("--titok_model_path", type=str, default="")
    parser.add_argument("--vqgan_model_path", type=str, default="")
    parser.add_argument("--gpu_str", type=str)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--ignore_keys", type=str, default="", help="split by +")
    return parser


def upload_trainingImage_data(data_src_dir, data_tar_dir, data_src_extra_dir=None):
    # make folder
    dst_folder = os.path.join(data_tar_dir, "open_images_train0123456")
    os.makedirs(dst_folder, exist_ok=True)

    # unzip images
    files = os.listdir(data_src_dir)
    for f in files:
        if f.endswith('.zip'):
            src_pack = os.path.join(data_src_dir, f)
            command = f"unzip -q -j {src_pack} -d {dst_folder}"
            run_command(command)

    # unzip extra images
    if data_src_extra_dir:
        files = os.listdir(data_src_extra_dir)
        for f in files:
            if f.endswith('.zip'):
                src_pack = os.path.join(data_src_extra_dir, f)
                command = f"unzip -q -j {src_pack} -d {dst_folder}"
                run_command(command)

    img_list_file = os.path.join(data_tar_dir, "open_images_train0123456.txt")
    find_images(dst_folder, img_list_file)

    return dst_folder, img_list_file


def upload_val_data(data_src_file, data_tar_dir):
    # make folder
    dst_folder = os.path.join(data_tar_dir, "val_images")
    os.makedirs(dst_folder, exist_ok=True)

    # unzip images
    command = f"unzip -q -j {data_src_file} -d {dst_folder}"
    run_command(command)

    img_list_file = os.path.join(data_tar_dir, "val_images.txt")
    find_images(dst_folder, img_list_file)

    return dst_folder, img_list_file


def upload_models(titok_dir, vqgan_dir, data_tar_dir):
    run_command(f"cp -r {titok_dir} {data_tar_dir}")
    run_command(f"cp -r {vqgan_dir} {data_tar_dir}")

    titok_tar_dir = os.path.join(data_tar_dir, os.path.split(titok_dir)[1])
    print_os(f"[DEBUG] copy {titok_dir} -> {titok_tar_dir}")
    titok_tar_file = os.path.join(titok_tar_dir, "tokenizer_titok_l32.bin")

    vqgan_tar_dir = os.path.join(data_tar_dir, os.path.split(vqgan_dir)[1])
    print_os(f"[DEBUG] copy {vqgan_dir} -> {vqgan_tar_dir}")
    vqgan_tar_file = os.path.join(vqgan_tar_dir, "checkpoints", "last.ckpt")

    return titok_tar_file, vqgan_tar_file


def main(opt):
    # 1. check work directory
    print_os(f"[DEBUG] current work directory is {os.getcwd()}")

    # 2. get folders
    print_os(f"[DEBUG] output_folder: {opt.out_dir}")
    print_os(f"[DEBUG] train dataset dir: {opt.train_dataset_dir}, val dataset dir: {opt.val_dataset_dir}")
    print_os(f"[DEBUG] titok model dir: {opt.titok_model_path}, vqgan model dir: {opt.vqgan_model_path}")
    print_os(f"[DEBUG] experiment_name: {opt.name}, config file: {opt.config}")
    run_command("pwd")
    run_command("ls")

    # 3. install dependencies
    print_os(f"[DEBUG] now installing dependencies...")
    install_dependency()
    print_os(f"[DEBUG] Finish installing dependencies!")
    
    # 4. create data folder
    run_command('df -h')
    base_dir = get_dataset_dst_folder()
    print_os(f"[DEBUG] get the base storage dir: {base_dir}")
    # make dataset dir
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    # make pretrain model dir
    pretrain_dir = os.path.join(base_dir, "pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)

    # 5. upload dataset
    _, train_image_file = upload_trainingImage_data(opt.train_dataset_dir, dataset_dir, opt.train_dataset_extra_dir)
    _, val_image_file = upload_val_data(opt.val_dataset_dir, dataset_dir)
    print_os(f"[DEBUG] train image file: {train_image_file}")
    print_os(f"[DEBUG] val image file: {val_image_file}")
    run_command('df -h')

    # 6. rebase config file
    from omegaconf import OmegaConf
    src_config = OmegaConf.load(opt.config)
    src_config.model.params.titok_pretrain_path = opt.titok_model_path
    src_config.model.params.vqganconfig.ckpt_path = opt.vqgan_model_path
    src_config.data.params.train.params.training_images_list_file = train_image_file
    src_config.data.params.validation.params.test_images_list_file = val_image_file

    # 6.1 load ckpt if need
    if opt.ckpt_path != "":
        src_config.model.params.ckpt_path = opt.ckpt_path
        if opt.ignore_keys != "":
            src_config.model.params.ignore_keys = opt.ignore_keys.split('+')

    # 6.2 save config file
    OmegaConf.save(src_config, "config_this.yaml")

    # 7. auto resume
    resume_record_path = os.path.join(opt.out_dir, f"auto_resume_{opt.name}.yaml")
    print_os(f"[DEBUG] Auto resume Enabled! record is in {resume_record_path}")

    # 9. start training
    common_args = (f" python ./train.py "
                   f" --base config_this.yaml "
                   f" --outdir {opt.out_dir} "
                   f" --name {opt.name} "
                   f" --gpus {opt.gpu_str} "
                   f" --save_top_k 6 "
                   f" --train "
                   f" --no_test "
                   f" --bar_rate 1000 "
                   f" --log_rate 200 "
                   f" --auto_resume y ")
    run_command(common_args)
    return 0


if __name__ == "__main__":
    print(f"[DEBUG] given args: {sys.argv}")
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    main(opt)
