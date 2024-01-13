conda activate sdwebui
CUDA_VISIBLE_DEVICES=0 python ./webui.py --port 7861 --share --xformers --no-half-vae --disable-nan-check --gradio-auth jiqixuexi:jiqixuexi --listen --disable-safe-unpickle
