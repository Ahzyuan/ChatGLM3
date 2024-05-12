eval "$(conda shell.bash hook)"
conda activate hzy

cd /data/hzy/ChatGLM3/task3

for i in {1..10}
do
  python main.py
done