import json
import os
import time
import zipfile
from multiprocessing import Pool
from shutil import copyfile


def worker(input_command):
    print(input_command)
    os.system(input_command)


def submit_commands(commands):
    with Pool(len(commands)) as p:
        p.map(worker, commands)


def get_args(argv):
    print(argv)
    working_folder = argv[1]
    dataset_folder = argv[2]
    experiment_name = argv[3]
    print(f"working_folder {working_folder}")
    print(f"dataset_folder {dataset_folder}")
    print(f"experiment_name {experiment_name}")
    return working_folder, dataset_folder, experiment_name


def install_dependency():
    os.system('pwd')
    os.system('ls')
    os.system('python -m pip install -U pip')
    os.system('python -m pip install -r requirements.txt')
    os.system('nvidia-smi')


def unzip_dataset(src_folder, dst_folder):
    print(f"unzipping from {src_folder} to {dst_folder}")
    for f in os.listdir(src_folder):
        if not f.endswith('.zip'):
            continue
        src_path = os.path.join(src_folder, f)
        with zipfile.ZipFile(src_path, 'r') as zip_ref:
            zip_ref.extractall(dst_folder)
        print(f"{time.ctime()} extracted {f}")


def upload_one_dataset(src_root, src_description, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    src_path = os.path.join(src_root, src_description)
    dst_path = os.path.join(dst_folder, "description.json")
    copyfile(src_path, dst_path)

    with open(src_path) as json_file:
        datasets = json.load(json_file)
    for dataset in datasets:
        dataset_name = dataset['dataset_name']
        src_folder = os.path.join(src_root, dataset_name)
        unzip_dataset(src_folder, dst_folder)


def upload_dataset(cluster='9.99'):
    if cluster == '9.99':
        train_dataset_dst_folder = '/gdata/shengxh/vimeo/vimeo_train'
        test_dataset_dst_folder = '/gdata/shengxh/vimeo/vimeo_test'
    else:
        train_dataset_dst_folder = '/data/liyao/vimeo/video_train'
        test_dataset_dst_folder = '/data/liyao/vimeo/vimeo_test'
    return train_dataset_dst_folder, test_dataset_dst_folder


def get_pretrained_weights(train_with_msssim=False):
    if train_with_msssim:
        image_models = '/gdata2/tangcb/benchmark/old_model/Iframe/cvpr2023_image_psnr.pth.tar'
    else:
        image_models = '/gdata2/tangcb/benchmark/old_model/Iframe/cvpr2023_image_psnr.pth.tar'
    me_net_path = "/data/1339417445/DMC/benchmark/spynet_finetune/cur_0622_FA_t04_t2_epo_5.pth"
    # me_net_path = "/gdata2/tangcb/benchmark/old_model/spy_our_tune/flow_pretrain_np/"

    return image_models, me_net_path


def config2command(config, flag='train'):
    if flag=='train':
        output_path = os.path.join(config['output_root'], config['experiment_name'])
        common_args = (f" python {config['train_file_path']}"
                       f" --train_dataset {config['train_path']} --train_batch_size {config['train_batch_size']}  --train_patch_size 256 256"
                       " --train_frame_num 6  --train_frame_selection random"
                       " --train_max_frame_distance 6 --train_random_flip 1"
                       " --train_min_zoom_factor 1.0 --train_max_zoom_factor 1.0"
                       f" --i_frame_model_path {config['image_model']}"
                       f" --me_net_pretrain_path {config['me_net_path']}"
                       f" -n 4 --epochs {config['epochs']} --training_scheduling {config['training_scheduling']}"
                       " --num_epoch_per_checkpoint 2 --load_ckpt 1"
                       f" --benchmark_test_anchor {os.path.join(config['benchmark_folder'], 'anchor', config['anchor_json'])}"
                       f" --benchmark_test_epoch {config['benchmark_test_epoch']}"
                       f" --benchmark_data_path {config['test_path']}"
                       f" --benchmark_test_config {config['benchmark_test_config']}")
        submit_args = f"{common_args}" \
                      f" --save_dir {output_path}" \
                      f" --model_name {config['model_name']}" \
                      f" --task_id 0" \
                      f" --lmbdas {config['lmbdas']}" \
                      f" --weights {config['weights']}" \
                      f" --cuda_idx {config['cuda_idx']}" \
                      f" --split_rate 1"
    elif flag=='test':
        output_json_path = f"/output/{config['experiment_name']}.json"
        command_line = (f" python {config['test_file_path']}"
                        f" --i_frame_model_path  {config['image_model']}"
                        f" --model_name {config['model_name']}"
                        f" --p_frame_model_path {config['video_model']}"
                        # " --rate_num 20"
                        # " --i_frame_q_scales 2.6 1.44 0.8 0.3"
                        # f" --p_frame_mv_y_q_scales 1.184 1.104 1.011 0.919"
                        # f" --p_frame_y_q_scales 1.238 0.962 0.713 0.532"
                        # " --force_intra 1"
                        f" --rate_num {config['rate_num']}"
                        f" --yuv420 {config['yuv420']}"
                        f" --test_config {config['test_config_json']} --cuda 1 -w {config['worker_num']}"
                        f" --output_path {output_json_path} --save_decoded_frame {config['save_decoded_frame']} --decoded_frame_path /model/Tombobo/Austin_Tang/Vcip/important_things/decoded_frame"
                        f" --write_stream {config['write_stream']} --stream_path /model/Tombobo/Austin_Tang/Vcip/important_things/bin")
        submit_args = command_line
    else: #"online"
        output_json_path = f"/output/{config['experiment_name']}.json"
        command_line = (f" python {config['test_file_path']}"
                        f" --i_frame_model_path  {config['image_model']}"
                        f" --model_name {config['model_name']}"
                        f" --p_frame_model_path {config['video_model']}"
                        # " --rate_num 20"
                        # " --i_frame_q_scales 2.6 1.44 0.8 0.3"
                        # f" --p_frame_mv_y_q_scales 1.184 1.104 1.011 0.919"
                        # f" --p_frame_y_q_scales 1.238 0.962 0.713 0.532"
                        # " --force_intra 1"
                        # f" -lr 5e-3 --update_times 1000"
                        # f" --inter_lmbda 840"
                        f" --rate_num {config['rate_num']}"
                        f" --yuv420 {config['yuv420']}"
                        f" --test_config {config['test_config_json']} --cuda 1 -w {config['worker_num']}"
                        f" --output_path {output_json_path} --save_decoded_frame {config['save_decoded_frame']} --decoded_frame_path /output/decoded_frame/"
                        f" --write_stream {config['write_stream']} --stream_path /output/bitstream/")
        submit_args = command_line

    return submit_args
