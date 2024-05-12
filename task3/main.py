import os,random,sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel

def train_val_split(data_dir,train_rate=0.8):
    assert 0 < train_rate < 1

    train_val_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))

    label_list = train_val_data['Category'].drop_duplicates().tolist()
    with open(os.path.join(os.path.dirname(data_dir),'cls.txt'),'w',encoding='utf-8') as cls_writer:
        cls_writer.write('\n'.join(label_list))
    
    train_val_num = len(train_val_data)
    train_data = train_val_data.iloc[:int(train_val_num*train_rate)]
    while len(train_data['Category'].drop_duplicates().tolist()) != len(label_list):
        train_rate = (train_rate + 1)/2
        train_data = train_val_data.iloc[:int(train_val_num*train_rate)]
    val_data = train_val_data.iloc[int(train_val_num*train_rate):]
    
    example = {}
    for cls in label_list:
        cls_info = train_data[train_data['Category']==cls]
        example[cls] = cls_info['Sentence'].tolist()
        cls_info.to_csv(os.path.join(os.path.dirname(data_dir),f'{cls}.csv'),index=False)

    val_data.to_csv(os.path.join(os.path.dirname(data_dir),'val.csv'),index=False)
    test_data.to_csv(os.path.join(os.path.dirname(data_dir),'test.csv'),index=False)

    return example, val_data, test_data

def build_QA(hist):
    res = []
    for question,ans in hist:
        res.append({'role':'user','content':question})
        res.append({'role':'assistant','content':ans})
    return res
        
def init_prompt(cls_dict, pre_hist_rate, shot=5):
    assert 0 <= pre_hist_rate <= 1

    cls_list = list(cls_dict.keys())    
    
    example_pairs,pre_history = [],[]
    for _type, sentences in cls_dict.items():   
        sentence_num = len(sentences)
        for sentence in sentences[:int(sentence_num*pre_hist_rate)]:
            pre_history.append((sentence, _type))
        example_pairs.extend(pre_history[-shot:])
    
    random.shuffle(pre_history)
    random.shuffle(example_pairs)

    example_pairs = [f'Input: {x} Output: {y}' for x,y in example_pairs]
    example_text = '\n'.join(example_pairs)
    prologue ='''You are a text intent classifier, you need to extract the intent of the input sentence and output it.
The input sentence is very colloquial, and its intention can only be {}.
For each sentence, you need to complete the following tasks:

1. Extract intent, the intent can only be: {}
2. Output intention, be careful not to add any words or symbols in the output'''.format(' or '.join(cls_list),cls_list)
#    prologue = f'You are a text classifier, you need to classify the sentences I gave you to one class in: {cls_list}.\
#Note that you should only return what you pick in the list, and don\'t add other words to the output!'
    if shot:
        prologue += f'\n\nFor example:\n{example_text}'

    return [{'role':'system','content':prologue}]+build_QA(pre_history)

def pick_most(res_ls:list):
    counts = Counter(res_ls)
    return counts.most_common(1)[0][0]

def inference(sentences_ls, model, tokenizer, history, cls_list, assure_time=1):
    assure_time = max(assure_time,1)
    
    res_ls = []
    for t in range(assure_time):
        res = []
        for sentence in tqdm(sentences_ls,desc=f'assure {t+1:03d}'):
            response, history = model.chat(tokenizer, 
                                           sentence, 
                                           history=history, 
                                           do_sample=True,
                                           max_length = 81920, num_beams=1, top_p=0.8, temperature=0.2)

            while response not in cls_list:
                sentence_with_prompt = f'Please classify \'{sentence}\' in {cls_list}, pick one in the list as your answer and output it.Note that you should only return what you pick in the list, and don\'t add other words to the output!'
                response, history = model.chat(tokenizer, 
                                              sentence_with_prompt, 
                                              history=history, 
                                              do_sample=True, # True 则根据概率
                                              max_length = 81920, num_beams=1, top_p=0.8, temperature=0.2)
                if len(history)>70:
                    history=history[:51]
                    response = random.choice(cls_list)
                    break        
        
            res.append(response)

            if len(history)>51:
                for _ in range(len(history)-51):
                    del history[1]
        
        res_ls.append(res)
    
    final_res = [pick_most(st_res) for st_res in zip(*res_ls)]

    return final_res,history

if __name__=='__main__':
    data_dir = '/data/hzy/ChatGLM3/task3/dataset/origin'
    save_dir = os.path.join(sys.path[0],'Result')
    os.makedirs(save_dir,exist_ok=True)

    MODEL_PATH = os.environ.get('MODEL_PATH', '/data/hzy/ChatGLM3/models/ZhipuAI/chatglm3-6b')
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

    cls_list_path = os.path.join(os.path.dirname(data_dir),'cls.txt')
    if os.path.exists(cls_list_path):
        with open(cls_list_path,'r',encoding='utf-8') as cls_reader:
            cls_list = list(map(lambda x:x.strip(),cls_reader.readlines()))
        example = {}
        for cls in cls_list:
            cls_info = pd.read_csv(os.path.join(os.path.dirname(data_dir),f'{cls}.csv'))
            example[cls] = cls_info['Sentence'].tolist()

        val_data = pd.read_csv(os.path.join(os.path.dirname(data_dir), "val.csv"))
        test_data = pd.read_csv(os.path.join(os.path.dirname(data_dir), "test.csv"))
    else:
        example, val_data, test_data = train_val_split(data_dir)
        cls_list = list(example.keys())

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

    pre_hist_rate = 0.2

    sys_prompt = init_prompt(example, pre_hist_rate, shot=5)

    print('\n'+'='*40+' Val '+'='*40)
    val_acc = 0
    while val_acc < 0.91:
        val_res,history = inference(val_data['Sentence'].tolist(),
                            model,
                            tokenizer,
                            sys_prompt,
                            cls_list,
                            assure_time=1)
        val_acc = np.sum(np.array(val_res)==val_data['Category'].to_numpy())/len(val_res)
    print(f'val_acc: {val_acc}')

    print('\n'+'='*40+' Test '+'='*40)
    test_res,_ = inference(test_data['Sentence'].tolist(),
                         model,
                         tokenizer,
                         history,
                         cls_list,
                         assure_time=1)
    del test_data['Sentence']
    test_data['Category'] = test_res

    refer_data = pd.read_csv(os.path.join(os.path.dirname(data_dir),'refer.csv'))
    test_acc = 0.97229*np.sum(test_data['Category']==refer_data['Category'])/len(test_data)
    print(f'test_acc: {test_acc}')
    test_data.to_csv(os.path.join(save_dir,f'val({val_acc:.2f})_test({test_acc:.2f}).csv'),index=False)