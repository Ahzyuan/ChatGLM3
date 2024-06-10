eval "$(conda shell.bash hook)"
conda activate hzy

cd /data/hzy/ChatGLM3/task4

# finetune
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python finetune_hf.py  /data/hzy/ChatGLM3/task4/dataset  /data/hzy/ChatGLM3/models/ZhipuAI/chatglm3-6b  /data/hzy/ChatGLM3/task4/configs/lora.yaml
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python finetune_hf.py  /data/hzy/ChatGLM3/task4/dataset  /data/hzy/ChatGLM3/models/ZhipuAI/chatglm3-6b  /data/hzy/ChatGLM3/task4/configs/ptuning_v2.yaml

# inference
#python infer_main.py --model_path /data/hzy/ChatGLM3/models/lora_model/checkpoint-200
#python infer_main.py --model_path /data/hzy/ChatGLM3/models/pv2_model/checkpoint-400 