import os,random,sys,re,torch
import pandas as pd
import pickle as pk
from tqdm import tqdm
from glob import glob
from torch import cosine_similarity as cos_sim
from preprocess import split_data
from transformers import AutoTokenizer, AutoModel

def build_QA(hist):
    res = []
    for question,ans in hist:
        res.append({'role':'user','content':question})
        res.append({'role':'assistant','content':ans})
    return res
        
def init_prompt(example, pre_hist_rate, shot=5):
    assert 0 <= pre_hist_rate <= 1

    random.shuffle(example)

    prologue ='''你是一位身经百战的求职高手，你需要从职位描述与要求中提取总结出岗位需要的关键技能。
你将接受一段岗位描述或岗位要求，并完成以下任务：
1. 过滤无用或意义空泛的语句，如“岗位要求”、“岗位描述”、“具有良好沟通能力”、“具有良好团队协作能力”等。
2. 从剩余语句中提取所需技能，如“掌握pytorch或tensorflow等深度学习框架”可提取为“pytorch或tensorflow”。
3. 将所提取的内容通过", "拼接，输出。'''
    if shot:
        example_pairs = random.sample(example,shot)
        example_pairs = [f'输入: {x} 输出: {y}' for x,y in example_pairs]
        example_text = '\n'.join(example_pairs)
        prologue += f'\n\n例如：\n{example_text}'

    return [{'role':'system','content':prologue}]+build_QA(example)

def clean_text(text):
    rules = [
        {r'\s+': u''},  # replace consecutive spaces
        {r'^\s+': u''},  # remove spaces at the beginning
        {r'[^\w\s]': u''} # remove punctuation
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.rstrip()
    return text.lower()

def inference(sentences_ls, model, tokenizer, history, assure_time=1):
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
        
            res.append(response)

            if len(history)>35:
                for _ in range(len(history)-36):
                    del history[1]
        
        res_ls.append(res)
    
    final_res = res_ls[0]#[pick_most(st_res) for st_res in zip(*res_ls)]

    return final_res,history

def sentence2vector(s,tokenizer, model):
    ipt = tokenizer(s,return_tensors='pt')
    opt = model(**ipt)
    return opt['logits'].mean(dim=1) #1,65024

if __name__=='__main__':
    data_dir = '/home/hzy/project/ChatGLM3/jobbing_skills/dataset'
    save_dir = os.path.join(sys.path[0],'Result')
    os.makedirs(save_dir,exist_ok=True)

    MODEL_PATH = os.environ.get('MODEL_PATH', '/home/hzy/project/ChatGLM3/models/ZhipuAI/chatglm3-6b-128k')
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

    purified_dir = os.path.join(data_dir, 'final_data')
    if os.path.exists(os.path.join(purified_dir,'example.pkl')):
        with open(os.path.join(purified_dir,'example.pkl'),'rb') as f:
            example = pk.load(f)
        test_data = []
        for csv_path in glob(purified_dir + '/*.xlsx'):
            test_data.append(pd.read_excel(csv_path))
        test_data = pd.concat(test_data)
    else:
        print(f'{purified_dir} not exists, please run preprocess.py and filt_irrelated_data.py first')
    example_ans = torch.cat([sentence2vector(res,tokenizer,model) for _, res in example],dim=0) #num, 65024

    sys_prompt = init_prompt(example, pre_hist_rate=0.5, shot=5)

    example_JD = [jd for jd,_ in example]

    print('\n'+'='*40+' Val '+'='*40)
    val_acc = 0
    while val_acc < 0.9:
        val_res,history = inference(example_JD,
                                    model,
                                    tokenizer,
                                    sys_prompt,
                                    assure_time=1)
        val_res = torch.cat([sentence2vector(res,tokenizer,model) for res in val_res],dim=0)
        val_acc = cos_sim(example_ans,val_res).mean()
        print(f'val_acc: {val_acc}')

    print('\n'+'='*40+' Test '+'='*40)
    test_res,_ = inference(test_data['JD'].tolist(),
                           model,
                           tokenizer,
                           history,
                           assure_time=1)
    test_data['技能'] = test_res

    test_data.to_excel(os.path.join(save_dir,f'val({val_acc:.2f}).xlsx'),index=False)