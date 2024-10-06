import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
os.environ['http_proxy'] = 'http://127.0.0.1:2345'
os.environ['https_proxy'] = 'http://127.0.0.1:2345'
os.environ['no_proxy'] = '127.0.0.1,localhost'

from model import *
from logger import *

import sys
import torch
from prettytable import PrettyTable
from transformers import AutoTokenizer

# we need this to import senteval
sys.path.insert(0, '../SentEval')
import senteval

sent_eval_mode = 'test'
tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']

data_path = '../SentEval/data'
best_model_path = './save/sj_best_model.pth'
model_path = 'jinaai/jina-embeddings-v2-base-en'
# 'jinaai/jina-embeddings-v2-base-en' # '../../models/nomic-embed-text-v1'

# n_cse_best_model.pth: 82.32
# j_cse_best_model.pth: 82.44

def show_table(task_names, scores):
    table = PrettyTable()
    table.field_names = task_names
    table.add_row(scores)
    print(table)

def main():
    if 'cse' in best_model_path:
        model = CSE_Model(model_path=model_path)
    elif 'bert' in model_path:
        model = Average_BERT(bert_path=model_path)
    else:
        model = Dual_Tower(model_path=model_path, regression=False)
    
    if os.path.exists(best_model_path):
        check_point = torch.load(best_model_path)
        model.load_state_dict(check_point['model'])  # corresponding to torch.save in train.py
        logger.info(f'load best model with epoch: {check_point["epoch"]} and dev score: {check_point["score"]}')
    else:
        raise ValueError(f'fail to load {best_model_path}')    
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    if sent_eval_mode == 'test':
        params = {
            'task_path': data_path,
            'usepytorch': True,
            'kfold': 10,
            'classifier': {
                'nhid': 0,
                'optim': 'adam',
                'batch_size': 64,
                'tenacity': 5,
                'epoch_size': 4,
            },
        }
    elif sent_eval_mode in ['dev', 'fasttest']:
        params = {
            'task_path': data_path,
            'usepytorch': True,
            'kfold': 5,
            'classifier': {
                'nhid': 0,
                'optim': 'rmsprop',
                'batch_size': 128,
                'tenacity': 3,
                'epoch_size': 2,
            },
        }
    else:
        raise ValueError(f'unknown {sent_eval_mode}')

    def prepare(params, samples):
        params.max_length = MAX_SEQUENCE_LENGTH
        return

    def batcher(params, batch):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in sentence] for sentence in batch]

        # batch is divided by token. we need to form sentences
        sentences = [' '.join(sentence) for sentence in batch]
        
        batch = tokenizer.batch_encode_plus(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=params.max_length,
            return_tensors='pt',
        )

        for i in batch:
            batch[i] = batch[i].to(device)

        with torch.no_grad():
            output = model.text2embedding(batch)
            
        return output.cpu()

    results = {}
    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    if sent_eval_mode == 'dev':
        print(f'------ {sent_eval_mode} ------')

        # STS
        scores = []
        task_names = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)

            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append('0.00')      

        show_table(task_names=task_names, scores=scores)

        # Transfer
        # scores = []
        # task_names = []
        # for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        #     task_names.append(task)
            
        #     if task in results:
        #         scores.append("%.2f" % (results[task]['acc']))
        #     else:
        #         scores.append('0.00')

        # task_names.append('Avg.')
        # scores.append("%.2f" % (sum(float(score) for score in scores) / len(scores)))
        # show_table(task_names=task_names, scores=scores)

    elif sent_eval_mode in ['test', 'fasttest']:
        print(f'------ {sent_eval_mode} ------')

        # STS
        scores = []
        task_names = []

        if 'STS22' not in tasks:
            for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
                task_names.append(task)

                if task in results:
                    if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                        scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                    else:
                        scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
                else:
                    scores.append('0.00')
        else:
            for task in ['STS22']:
                task_names.append(task)

                if task in results:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
                else:
                    scores.append('0.00')         

        task_names.append('Avg.')
        scores.append("%.2f" % (sum(float(score) for score in scores) / len(scores)))
        show_table(task_names=task_names, scores=scores)

        # Transfer
        # scores = []
        # task_names = []
        # for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        #     task_names.append(task)
            
        #     if task in results:
        #         scores.append("%.2f" % (results[task]['acc']))
        #     else:
        #         scores.append('0.00')

        # task_names.append('Avg.')
        # scores.append("%.2f" % (sum(float(score) for score in scores) / len(scores)))
        # show_table(task_names=task_names, scores=scores)

if __name__ == '__main__':
    main()    