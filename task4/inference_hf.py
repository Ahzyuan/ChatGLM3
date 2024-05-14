#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Annotated, Union
import pandas as pd
import numpy as np
import typer,os,sys
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
        data_base: Annotated[str, typer.Argument(help='')],
):
    test_data = pd.read_csv(os.path.join(data_base,'test.csv'))
    ref_data = pd.read_csv(os.path.join(data_base,'refer.csv'))
    sentences = test_data['Sentence'].tolist()

    model, tokenizer = load_model_and_tokenizer(model_dir)

    res = []
    for st in tqdm(sentences):
        response, _ = model.chat(tokenizer, st)
        res.append(response)

    del test_data['Sentence']
    test_data['Category'] = res

    test_acc = 0.97229*np.sum(test_data['Category']==ref_data['Category'])/len(test_data)
    print(f'test_acc: {test_acc}')
    
    save_dir = os.path.join(sys.path[0],'Result')
    os.makedirs(save_dir,exist_ok=True)
    test_data.to_csv(os.path.join(save_dir,f'test({test_acc:.2f}).csv'),index=False)


if __name__ == '__main__':
    app()
