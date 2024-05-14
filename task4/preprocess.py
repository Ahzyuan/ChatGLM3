import json,os,random,shutil
import pandas as pd

def build_QA(hist):
    '''
    hist: (question, answer)
    '''
    res = []
    for question,ans in hist:
        res.append([{'role':'user','content':question},{'role':'assistant','content':ans}])
    return res

def save_json(data, cls_list, save_path):
    sys_prompt = '''You are a text intent classifier, you need to extract the intent of the input sentence and output it.
The input sentence is very colloquial, and its intention can only be {}.
For each sentence, you need to complete the following tasks:

1. Extract intent, the intent can only be: {}
2. Output intention, be careful not to add any words or symbols in the output'''.format(' or '.join(cls_list),cls_list)
    
    #example_pairs = []
    #for cls in cls_list:
    #    cls_data = data[data['Category']==cls].head(5).values.tolist()
    #    example_pairs.extend([f'Input: {q} Output: {a}' for _,q,a in cls_data])
    #random.shuffle(example_pairs)
    #example_text = '\n'.join(example_pairs)
    #sys_prompt += f'\nFor example:\n{example_text}'
    sys_conv = [{"role":"system","content":sys_prompt}]

    qa_pairs = [row.tolist()[1:] for _,row in data.iterrows()]
    
    with open(save_path, 'wt', encoding='utf-8') as fout:
        res = []
        for conv in build_QA(qa_pairs):
            sample = {'conversations': sys_conv+conv}
            res.append(sample)
        fout.write(json.dumps(res, indent=4, ensure_ascii=False) + '\n')

if __name__=='__main__':
    data_base = '/data/hzy/ChatGLM3/task4/dataset/origin'
    save_dir = os.path.dirname(data_base)
    os.makedirs(save_dir,exist_ok=True)

    train_rate = 0.8

    train_val_data = pd.read_csv(os.path.join(data_base, "train.csv"))

    cls_list = train_val_data['Category'].drop_duplicates().tolist()
    
    train_val_nums = len(train_val_data)
    train_data = train_val_data.iloc[:int(train_val_nums*train_rate)]
    val_data = train_val_data.iloc[int(train_val_nums*train_rate):]

    save_json(train_data, cls_list, os.path.join(save_dir,'train.json'))
    save_json(val_data, cls_list, os.path.join(save_dir,'val.json'))
    shutil.copyfile(os.path.join(data_base,'test.csv'),
                    os.path.join(save_dir,'test.csv'))
    print('Done.')
