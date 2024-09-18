import os,re
import pandas as pd

xlsx_root = '/home/hzy/project/ChatGLM3/jobbing_skills/dataset/origin'
save_root = os.path.join(os.path.dirname(xlsx_root), 'merged_data')
os.makedirs(save_root, exist_ok=True)

pattern = re.compile('[^\d.]+')

type_dict = {}
for file in os.listdir(xlsx_root):
    file_name = file.split('.')[0]
    file_type = re.findall(pattern, file_name)[0]
    type_dict[file_type] = type_dict.get(file_type, []) + [file]

for type, files in type_dict.items():
    if len(files) < 2: continue
    type_data = [pd.read_excel(os.path.join(xlsx_root, file)) for file in files]
    type_data = pd.concat(type_data)
    type_data.to_excel(os.path.join(save_root, type + '.xlsx'), index=False)
    