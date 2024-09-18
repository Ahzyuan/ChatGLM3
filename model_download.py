import os
from modelscope.hub.snapshot_download import snapshot_download

# save_dir是模型保存到本地的目录
save_dir="/home/hzy/project/ChatGLM3/models"
os.makedirs(save_dir, exist_ok=True)

snapshot_download("ZhipuAI/chatglm3-6b-128k", 
                  cache_dir=save_dir)