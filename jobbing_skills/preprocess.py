import os,sys
import numpy as np
import pandas as pd
import pickle as pk
from collections import Counter

'''
This script is to unique the whole job data and split the data into train and test
'''

def unique_data(df):
    origin_num = len(df)

    judge_item = []
    empty_mask = np.array([False]*origin_num)
    for idx, (c,a,j) in enumerate(zip(df['公司'],df['地址'],df['岗位'])):
        if pd.isna(c) or pd.isna(a) or pd.isna(j):
            empty_mask[idx] = True
            continue
        judge_item.append(''.join([c,a,j]))
    df = df[~empty_mask]
    item_times = Counter(judge_item)
    judge_item = np.array(judge_item)

    multi_type_idx = set()
    del_mask = np.array([False]*len(df))
    for item, times in item_times.items():
        if times == 1:
            continue

        idx = np.where(judge_item == item)[0]
        final_type = [df.iloc[idx[0],0]]
        for i in idx[1:]:
            if df.iloc[i,0] not in final_type:
                final_type.append(df.iloc[i,0])  # 合并职业类型
                multi_type_idx.add(idx[0])
            del_mask[i] = True
        df.iloc[idx[0],0] = ', '.join(final_type)
    multi_type_data = df.iloc[list(multi_type_idx)]
    df = df[~del_mask]
    
    final_num = len(df)
    print(f'已去重：{origin_num} -> {final_num}')
    return multi_type_data, df

def split_data(data_dir, purify_dir):
    '''
    data_dir: contains several xlsx files, in which record one kind of job data
    '''
    # example: jd-extraction(split by ', ')
    example = pd.read_excel(os.path.join(data_dir,'example.xlsx')) # example.xlsx 含 JD 与技能两列
    example = [(jd, ans) for jd, ans in zip(example['JD'], example['技能'])]
    
    job_datas = []
    for file in os.listdir(data_dir):
        if file == 'example.xlsx':
            continue
        data = pd.read_excel(os.path.join(data_dir,file))
        data.insert(0,'类型',file.split('.')[0])
        job_datas.append(data)
    
    # test_data: Dataframe, column：类型，公司，链接，地址，岗位，经验，薪资，JD
    test_data = pd.concat(job_datas)
    job_type = test_data['类型'].unique()
    if '技能' in test_data:
        del test_data['技能']
    multi_type_data, test_data = unique_data(test_data)
    #test_data.to_excel(os.path.join(purify_dir, 'all.xlsx'),index=False)

    with open(os.path.join(purify_dir,'example.pkl'),'wb') as f:
        pk.dump(example,f)
    
    for type in job_type:
        type_mask = test_data['类型'] == type
        type_data = test_data.loc[type_mask]
        type_data.to_excel(os.path.join(purify_dir, f'{type}.xlsx'),index=False)
    if len(multi_type_data):
        multi_type_data.to_excel(os.path.join(purify_dir, 'multi_type.xlsx'),index=False)

    return example, test_data

if __name__ == '__main__':
    data_dir = '/home/hzy/project/ChatGLM3/jobbing_skills/dataset/merged_data'
    purify_dir = os.path.join(os.path.dirname(data_dir), 'uniqued_date')
    os.makedirs(purify_dir,exist_ok=True)
    
    example, test_data = split_data(data_dir, purify_dir)
