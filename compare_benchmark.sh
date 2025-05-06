python compare_rd_video.py \
    --compare_between class \
    --output_path stdout \
    --base_method VTM \
    --plot_scheme separate \
    --distortion_metrics psnr \
    --plot_path ./output/IP32 \
    --log_paths VTM /code/results/IP32/benchmark/RGB-PSNR/VTM_yuv444_IntraPeriod_32encoder_lowdelay_main_rext_psnr_msssim_DRT2_all_datasets.json \
    DCVC /code/results/IP32/benchmark/RGB-PSNR/DCVC-PSNR.json \
    DCVC-TCM /code/results/IP32/benchmark/RGB-PSNR/DCVC-TCM.json \
    DCVC-HEM /code/results/IP32/benchmark/RGB-PSNR/DCVC-HEM-RGB-PSNR.json \
    DCVC-DC /code/results/IP32/benchmark/RGB-PSNR/DCVC-DC-RGB-PSNR.json \
    'Sheng-2024' /code/results/IP32/benchmark/RGB-PSNR/DCVC-SDD-PSNR.json \
    DCVC-FM /code/results/IP32/benchmark/RGB-PSNR/DCVC-FM-RGB-PSNR-IP32.json \
    'SEVC (ours)' /code/results/IP32/PNG_RD/LBNVC_RGB_Final.json