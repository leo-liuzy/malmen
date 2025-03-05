export CUDA_VISIBLE_DEVICES=0
<<<<<<< HEAD
# python main.py data=zsre model=gpt-j editor=malmen editor.cache_dir=/data/users/zliu/malmen/cache

python edit.py data=zsre model=gpt-j editor=malmen editor.cache_dir=/data/users/zliu/malmen/cache editor.load_checkpoint=True
=======

python main.py data=musique_combiner_text model=llama3.2-1B-eos-sft data.outer_loop_include_atomq=True editor=malmen editor.cache_dir=cache/musique_combiner_text_w-atomq
>>>>>>> 5f1a7f65b99a9cf47aef6b13a56d426d9e6858aa
