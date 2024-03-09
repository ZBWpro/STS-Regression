from settings import *

import torch
from torch.utils.data import Dataset


class SiameseProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, item):
        query = item['sentence1']
        title = item['sentence2']
        label = float(item['gold_label'])

        # not specifying return_tensors here. we do not want to wrap an extra layer around the original list
        encoded_query = self.tokenizer(query, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH)
        encoded_title = self.tokenizer(title, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH)
        
        encoded_query = {  # list to tensor
            'input_ids' : torch.LongTensor(encoded_query['input_ids']).cuda(), 
            'attention_mask' : torch.LongTensor(encoded_query['attention_mask']).cuda()
        }

        encoded_title = {  # list to tensor
            'input_ids' : torch.LongTensor(encoded_title['input_ids']).cuda(), 
            'attention_mask' : torch.LongTensor(encoded_title['attention_mask']).cuda()
        }

        return encoded_query, encoded_title, label


class MyDataSet(Dataset):
    def __init__(self, tokenizer, data_set) -> None:
        super().__init__()
        self.data_set = data_set
        self.processor = SiameseProcessor(tokenizer)
    
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, index):
        data = self.data_set[index]
        return self.processor(data)