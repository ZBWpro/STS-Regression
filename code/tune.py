
import os
os.environ['NUMEXPR_MAX_THREADS'] = '88'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

from model import *
from logger import *
from data_set import *
from loss import TranslatedReLU, SmoothK2Loss

import sys
import random
import numpy as np
from datasets import load_dataset

from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup


'''
tune RoBERTa-base
    loss_function = SmoothK2Loss(threshold=0.25, k=3)
    source = 'sts-sick-round'
    bert_learing_rate = 4e-5
    epoch, batch_size = 5, 4
    evaluation_per_step = 50

tune BERT-base
    loss_function = SmoothK2Loss(threshold=0.25, k=4)
    source = 'sts-sick-round'
    bert_learing_rate = 5e-5
    epoch, batch_size = 5, 4
    evaluation_per_step = 50
'''

# path
save_path = './save'
best_model_path = './save/train-bert-sk2-76.03.pth'
bert_path = '../../../models/bert-base-uncased'

# hyper-parameter
source = 'sts-sick-round'
bert_learing_rate = 5e-5
epoch, batch_size = 5, 4
evaluation_per_step = 50

# we need this to import senteval
sys.path.insert(0, '../SentEval')
import senteval

evaluation_mode = 'dev'
evaluation_tasks = ['STSBenchmark']
sent_eval_data_path = '../SentEval/data'

# we use DDP to tune our model
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)


def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed()

    # prepare model
    model = Average_BERT(bert_path=bert_path)

    if os.path.exists(best_model_path):
        check_point = torch.load(best_model_path, map_location=device)
        model.load_state_dict(check_point['model'])  # corresponding to torch.save in train.py
        logger.info(f'load best model with epoch: {check_point["epoch"]} and dev score: {check_point["score"]}')

        best_score = check_point['score']
    else:
        raise ValueError(f'fail to load {best_model_path}')
    
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    # prepare data
    if source == 'sts-sick-round':
        tune_data = load_dataset('json', data_dir='../data', data_files='sts-sick-round.jsonl', cache_dir='./cache')
    else:
        raise ValueError(f'unknown data source : {source}')
    
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    tune_data_set = MyDataSet(tokenizer=tokenizer, data_set=tune_data['train'])

    tune_sampler = DistributedSampler(tune_data_set)
    tune_data_loader = DataLoader(dataset=tune_data_set, batch_size=batch_size * dist.get_world_size(), sampler=tune_sampler)

    # prepare optimizer
    layer_learning_rate = {}
    base_learning_rate = bert_learing_rate

    for i in range(11, -1, -1):
        layer_learning_rate[f'.{str(i)}.'] = base_learning_rate
        base_learning_rate *= RATE_DECAY_FACTOR

    optimizer_grouped_parameters = []
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]

    for name, parameter in parameters:
        params = {
            'params': [parameter],
            'lr': BERT_LEARNING_RATE,
            'weight_decay': 0.0
            if any(item in name for item in no_decay)
            else ADAM_WEIGHT_DECAY
        }   

        for layer_name, learning_rate in layer_learning_rate.items():
            if layer_name in name:
                params['lr'] = learning_rate
                break

        optimizer_grouped_parameters.append(params)
    
    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=BERT_LEARNING_RATE,
                      betas=(ADAM_BETA_1, ADAM_BETA_2),
                      eps=ADAM_EPSILON,
                      weight_decay=ADAM_WEIGHT_DECAY)
    
    # prepare scheduler
    total_steps = len(tune_data_set) * epoch // (batch_size * dist.get_world_size())
    warm_up_steps = int(total_steps * WARM_UP_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warm_up_steps, num_training_steps=total_steps)

    if local_rank == 0:
        logger.info(f'total steps: {total_steps}, with warm up steps: {warm_up_steps} and decay rate: {RATE_DECAY_FACTOR}')

    steps = 0
    current_steps = 0
    for e_i in range(epoch):
        for batch in tune_data_loader:
            dist.barrier()
            model.train()

            encoded_1, encoded_2 = batch[:2]
            prediction = model(encoded_1, encoded_2)

            # special treatment for the left critical point
            mask = (prediction >= 0).type(prediction.dtype)
            prediction = prediction * mask

            label = torch.FloatTensor(list(map(int, batch[-1]))).cuda()
            label = label.reshape(label.shape[0], 1)

            loss_function = SmoothK2Loss(threshold=0.25, k=4)
            loss = loss_function(prediction, label)
            loss.backward()

            clip_grad_norm_(parameters=model.parameters(), max_norm=GRADIENT_CLIPPING)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            dist.barrier()
            current_steps += dist.get_world_size()
            steps += dist.get_world_size()

            if steps < evaluation_per_step:
                continue
            
            steps = 0
            model.eval()

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

                for j in batch: 
                    batch[j] = batch[j].to(device)

                with torch.no_grad():
                    outputs = model.module.text2embedding(batch)

                return outputs.cpu()

            if evaluation_mode in ['dev', 'fasttest']:
                params = {
                    'task_path': sent_eval_data_path,
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
            elif evaluation_mode == 'test':
                params = {
                    'task_path': sent_eval_data_path,
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
            else:
                raise ValueError(f'unknown {evaluation_mode}')
        
            if local_rank == 0:
                total_score = 0
                for task in evaluation_tasks:
                    se = senteval.engine.SE(params, batcher, prepare)
                    result = se.eval(task)
                    total_score += result['dev']['spearman'][0] * 100
                
                average_score = total_score / len(evaluation_tasks)
                logger.info(f'epoch: {e_i}, steps: {current_steps}, proportion: {current_steps / total_steps}, score: {average_score}')
                
                if average_score > best_score:
                    best_score = average_score
                    logger.info(f'new best model, epoch: {e_i}, score: {best_score} !!!')
                    torch.save({'model': model.module.state_dict(), 'epoch': e_i, 'score': best_score}, open(os.path.join(save_path, 'tune_model.pth'), 'wb'))


if __name__ == '__main__':
    main()