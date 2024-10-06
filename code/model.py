from settings import *

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class Average_BERT(nn.Module):
    def __init__(self, bert_path) -> None:
        super(Average_BERT, self).__init__()
        self.config = AutoConfig.from_pretrained(bert_path, output_hidden_states=False)
        self.bert = AutoModel.from_pretrained(bert_path, config=self.config, add_pooling_layer=False)
        
        self.hidden_size = self.bert.config.hidden_size
        self.dense = nn.Linear(in_features=3 * self.hidden_size, out_features=1)
    
    def text2embedding(self, encoded_text):
        output = self.bert(**encoded_text)
        attention_mask = encoded_text['attention_mask']
        last_hidden_state = output.last_hidden_state
        return ((last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

    def forward(self, encoded_query, encoded_title):
        query_embedding = self.text2embedding(encoded_query)
        title_embedding = self.text2embedding(encoded_title)
        difference_embedding = abs(query_embedding - title_embedding)
        final_embedding = torch.cat([query_embedding, title_embedding, difference_embedding], dim=1)
        return self.dense(final_embedding)


class Dual_Tower(nn.Module):
    def __init__(self, model_path) -> None:
        super(Dual_Tower, self).__init__()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = self.model.config.hidden_size
        self.dense = nn.Linear(in_features=3 * self.hidden_size, out_features=1)
    
    def text2embedding(self, encoded_text):
        output = self.model(**encoded_text)
        attention_mask = encoded_text['attention_mask']
        last_hidden_state = output.last_hidden_state
        return ((last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

    def forward(self, encoded_query, encoded_title):
        query_embedding = self.text2embedding(encoded_query)
        title_embedding = self.text2embedding(encoded_title)
        difference_embedding = abs(query_embedding - title_embedding)
        final_embedding = torch.cat([query_embedding, title_embedding, difference_embedding], dim=1)
        return self.dense(final_embedding)


