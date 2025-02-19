CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node=2 train_lavis.py --cfg-path lavis/projects/abnclip/train/pretrain_stage1.yaml