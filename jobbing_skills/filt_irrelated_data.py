import rich,os,time,sys
import pandas as pd
from rich import print
from rich.text import Text
from rich.panel import Panel
from rich.console import Console

xlsx_root = '/home/hzy/project/ChatGLM3/jobbing_skills/dataset/uniqued_data'
save_dir = os.path.join(os.path.dirname(xlsx_root), 'final_data')
os.makedirs(save_dir, exist_ok=True)

console = Console()

def render(progress, job, jd):
    print('[bold][red]'+'-'*console.width+'[/red][/bold]') 
    print(Panel(Text(f'{progress}\t{job}', justify="center", style='bold green italic'), 
                     box=rich.box.DOUBLE,
                     border_style='green'))
    print(jd)

def printer_effect(text, delay=0.05):
    for char in text:
        console.print(char, end='',style='cyan bold')
        time.sleep(delay)
    print() 

for xlsx_file in os.listdir(xlsx_root):
    if not xlsx_file.endswith('.xlsx'): 
        if xlsx_file.endswith('.pkl'): # example.pkl
            shutil.move(os.path.join(xlsx_root, xlsx_file), 
                        os.path.join(save_dir, xlsx_file))
        continue
    file_path = os.path.join(xlsx_root, xlsx_file)
    df = pd.read_excel(file_path)
    save_idx = []
    all_finish = False

    try:
        for data_id, (job, jd) in enumerate(zip(df['岗位'],df['JD'])):
            render(f'{data_id}/{len(df)}', job, jd) # render markdown
            print('[bold][magenta]\nDelete [any key (not enter) to delete, enter to save]? > [/magenta][/bold]',end='')
            if_drop = input()
            if not if_drop:
                save_idx.append(data_id)
            elif 'back' in if_drop.lower():
                if data_id -1 in save_idx:
                    save_idx.pop(-1)
                else:
                    save_idx.append(data_id-1)
                
                if if_drop.lower() == 'back': # back and save present item
                    save_idx.append(data_id)
                else: # back and not save present item
                    pass
        printer_effect(f'\nCongratulation! {xlsx_file} finish...')
        all_finish = True

    except KeyboardInterrupt:
        left_path = os.path.join(xlsx_root, xlsx_file)
        printer_effect(f'\nSaving picked data items to {left_path}...')
        left_data = df.iloc[data_id:]
        left_data.to_excel(left_path, index=False) # save left data  
        break
    
    finally:
        if not save_idx: sys.exit(0)
        df = df.iloc[save_idx]
        save_path  = os.path.join(save_dir, xlsx_file) # save picked data 
        if os.path.exists(save_path): # add to last picked data
            df = pd.concat([pd.read_excel(save_path), df])
        df.to_excel(save_path, index=False)

        if all_finish:
            os.remove(file_path)
