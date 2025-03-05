export CUDA_VISIBLE_DEVICES=0

python main.py data=musique_combiner_text model=llama3.2-1B-eos-sft data.outer_loop_include_atomq=True editor=malmen editor.cache_dir=cache/musique_combiner_text_w-atomq