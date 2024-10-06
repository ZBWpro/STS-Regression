from model import *
from logger import *

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

import sys
from prettytable import PrettyTable
from transformers import AutoTokenizer

# to import senteval, we need this
sys.path.insert(0, '../SentEval')
import senteval

sent_eval_mode = 'test'
tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
# tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

# env: test-sts
data_path = '../SentEval/data'

# './save/tune-roberta-sk2.pth' => 83.23  './save/tune-bert-sk2.pth' => 82.93
best_model_path = './save/bert_train.pth'
bert_path = '../../models/bert-base-uncased'
#'../../models/roberta-base' # '../../models/bert-base-uncased'


def show_table(task_names, scores):
    table = PrettyTable()
    table.field_names = task_names
    table.add_row(scores)
    print(table)
    

def main():
    model = Average_BERT(bert_path=bert_path)

    if os.path.exists(best_model_path):
        check_point = torch.load(best_model_path)
        model.load_state_dict(check_point['model'])  # corresponding to torch.save in train.py
        logger.info(f'load best model with epoch: {check_point["epoch"]} and dev score: {check_point["score"]}')
    else:
        raise ValueError(f'fail to load {best_model_path}')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=bert_path)
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
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)

            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
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