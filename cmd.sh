CUDA_VISIBLE_DEVICES=0 python inference/real3d_infer.py \
--src_img person1.png \
--drv_aud aud1.mp3 \
--drv_pose static \
--bg_img bg1.png \
--out_name person1_bg1_aud1.mp4 \
--out_mode final \
--low_memory_usage

CUDA_VISIBLE_DEVICES=0 python inference/real3d_infer.py \
--src_img person1.png \
--drv_aud boyin_1min.wav \
--drv_pose data/raw/examples/1.mp4 \
--bg_img bg1.png \
--out_name id1_bg1_1min.mp4 \
--out_mode final \
--low_memory_usage

