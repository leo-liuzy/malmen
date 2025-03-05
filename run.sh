export CUDA_VISIBLE_DEVICES=3

python main.py data=musique_combiner_text model=llama3.2-1B-eos-sft data.outer_loop_include_atomq=True editor=malmen editor.cache_dir=cache/musique_combiner_text_w-atomq checkpoint_save_dir=checkpoints/musique_combiner_text_w-atomq

python main.py data=musique_combiner_text model=llama3.2-1B-eos-sft data.outer_loop_include_atomq=False editor=malmen editor.cache_dir=cache/musique_combiner_text checkpoint_save_dir=checkpoints/musique_combiner_text