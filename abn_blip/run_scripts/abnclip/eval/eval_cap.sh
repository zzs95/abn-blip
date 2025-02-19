CUDA_VISIBLE_DEVICES=1 /home/brownradx/miniconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=1 evaluate_lavis.py --cfg-path lavis/projects/abnclip/eval/caption_eval.yaml
# using 1 card for inference


