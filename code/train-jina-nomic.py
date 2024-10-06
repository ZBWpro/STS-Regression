import os
os.environ['NUMEXPR_MAX_THREADS'] = '88'
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

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


# path
save_path = './save'
model_path = 'jinaai/jina-embeddings-v2-base-en'
# 'jinaai/jina-embeddings-v2-base-en' # '../../models/nomic-embed-text-v1'

# we need this to import senteval
sys.path.insert(0, '../SentEval')
import senteval

evaluation_mode = 'dev'
evaluation_tasks = ['STSBenchmark']
sent_eval_data_path = '../SentEval/data'

# we use DDP to train our model
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
    
    BATCH_SIZE = 8
    EVALUATION_PER_STEP = 10000

    # {'train': ['gold_label', 'sentence1', 'sentence2']}
    train_data = load_dataset('csv', data_dir='../data', data_files='nli_012.csv', cache_dir='./cache')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_data_set = MyDataSet(tokenizer=tokenizer, data_set=train_data['train'])

    train_sampler = DistributedSampler(train_data_set)
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=BATCH_SIZE * dist.get_world_size(), sampler=train_sampler)

    model = Dual_Tower(model_path=model_path)
    
    for param in model.model.parameters():
        param.requires_grad = False
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=BERT_LEARNING_RATE,
                      betas=(ADAM_BETA_1, ADAM_BETA_2),
                      eps=ADAM_EPSILON,
                      weight_decay=ADAM_WEIGHT_DECAY)
    
    if local_rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}")
    
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    total_steps = len(train_data_set) * EPOCH // (BATCH_SIZE * dist.get_world_size())
    warm_up_steps = int(total_steps * WARM_UP_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warm_up_steps, num_training_steps=total_steps)

    if local_rank == 0:
        logger.info(f'total steps: {total_steps}, with warm up steps: {warm_up_steps} and decay rate: {RATE_DECAY_FACTOR}')

    steps = 0
    best_score = 0
    current_steps = 0
    for e_i in range(EPOCH):
        for batch in train_data_loader:
            dist.barrier()
            model.train()

            encoded_1, encoded_2 = batch[:2]
            prediction = model(encoded_1, encoded_2)

            # special treatment for the left critical point
            mask = (prediction >= 0).type(prediction.dtype)
            prediction = prediction * mask
            
            label = torch.FloatTensor(list(map(int, batch[-1]))).cuda()
            label = label.reshape(label.shape[0], 1)
            
            if 'jina' in model_path:
                loss_function = SmoothK2Loss(threshold=0.2, k=2)
            elif 'nomic' in model_path:
                loss_function = SmoothK2Loss(threshold=0.25, k=2)
            else:
                raise ValueError('unknown model path')
                
            loss = loss_function(prediction, label)
            loss.backward()

            clip_grad_norm_(parameters=model.parameters(), max_norm=GRADIENT_CLIPPING)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            dist.barrier()
            current_steps += dist.get_world_size()
            steps += dist.get_world_size()

            if steps < EVALUATION_PER_STEP:
                continue

            steps = 0
            model.eval()
            
            if local_rank == 0:
                logger.info(f'save new dense layer !!!')
                logger.info(f'epoch: {e_i}, steps: {current_steps}, proportion: {current_steps / total_steps}')
                
                if 'jina' in model_path:
                    torch.save({'model': model.module.state_dict(), 'epoch': e_i, 'score': best_score}, open(os.path.join(save_path, f'jina_frozen_train.pth'), 'wb'))
                elif 'nomic' in model_path:
                    torch.save({'model': model.module.state_dict(), 'epoch': e_i, 'score': best_score}, open(os.path.join(save_path, f'nomic_frozen_train.pth'), 'wb'))
                else:
                    raise ValueError('unknown model path')


if __name__ == '__main__':
    main()