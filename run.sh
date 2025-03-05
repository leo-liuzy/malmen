export CUDA_VISIBLE_DEVICES=0
# python main.py data=zsre model=gpt-j editor=malmen editor.cache_dir=/data/users/zliu/malmen/cache

python edit.py data=zsre model=gpt-j editor=malmen editor.cache_dir=/data/users/zliu/malmen/cache editor.load_checkpoint=True