import os

# ----------------------------Test-------------------------------
config_test = {
    "image_model": "/data/Tombobo/DCVC_bo/model/Iframe/cvpr2023_image_psnr.pth.tar",
    "video_model": "/model/EsakaK/My_Model/SEVC/SEVC_RGB_Final.pth.tar",
    "test_config_json": "config_F96-IP-1.json",
    "yuv420": 0,
    "save_decoded_frame": 0,
    "write_stream": 0,
    "ratio": 4.0,
    "worker_num": 8,
    "rate_num": 4,
    "test_file_path": "test.py",
    "experiment_name": f"IP-32-F96"
}

output_json_path = f"/output/{config_test['experiment_name']}.json"
test_command = (f" python {config_test['test_file_path']}"
                f" --i_frame_model_path  {config_test['image_model']}"
                f" --p_frame_model_path {config_test['video_model']}"
                f" --rate_num {config_test['rate_num']}"
                f" --yuv420 {config_test['yuv420']} --ratio {config_test['ratio']}"
                f" --test_config {config_test['test_config_json']} --cuda 1 -w {config_test['worker_num']} --verbose 2"
                f" --output_path {output_json_path} --save_decoded_frame {config_test['save_decoded_frame']} --decoded_frame_path /data/EsakaK/output/SEVC"
                f" --write_stream {config_test['write_stream']} --stream_path /model/Tombobo/Austin_Tang/Vcip/important_things/bin")

# ----------------------submit test--------------------
print(test_command)
os.system(test_command)
