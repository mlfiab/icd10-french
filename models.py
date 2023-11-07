import torch
from transformers import AutoTokenizer, AutoModel, FlaubertTokenizer, FlaubertModel, AdamW # get_linear_schedule_with_warmup
import torch.nn as nn


class BERT_Hierarchical_Model_Laat(nn.Module):

    def __init__(self, nb_classes, model_name, hidden):
        super(BERT_Hierarchical_Model_Laat, self).__init__()

        #self.bert_path = 'bert-base-uncased'
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        self.l1 = nn.Linear(hidden, hidden, bias=False)
        self.l2 = nn.Linear(hidden, nb_classes, bias=False)
        self.out = nn.Linear(hidden, nb_classes)


    def forward(self, ids, mask, token_type_ids, lengt):
        #print("Inputs ", ids.shape)
        pooled_out = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        hidden_output = pooled_out['last_hidden_state']
        #pooled_output = pooled_out['pooler_output']

        chunks_emb = hidden_output.split_with_sizes(lengt)
        batch_emb_pad = nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=-91, batch_first=True)

        batch_size, num_chunks, chunk_size, dim = batch_emb_pad.size()
        hidden_output = batch_emb_pad.view(batch_size, num_chunks*chunk_size, -1)

        weights = torch.tanh(self.l1(hidden_output))
        att_weights = self.l2(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1,2)
        weighted_output = att_weights @ hidden_output
        logits = self.out.weight.mul(weighted_output).sum(dim=2).add(self.out.bias)

        return logits
    
class FlauBERTClass(nn.Module):
    def __init__(self, nb_classes, model_name):
        super(FlauBERTClass, self).__init__()
        self.l1 = FlaubertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, nb_classes)

    def forward(self, ids, mask, token_type_ids, lengt):
        pooled_out, = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids,  return_dict=False)
        # Redimensionner la sortie de Embedding 3D to 2D
        pooled_out = pooled_out.permute(0, 2, 1)[:, :, -1]
        # print(pooled_out.shape)
        # print(pooled_out)

        pooler = self.dropout(pooled_out)
        pooler = self.dropout(pooler)

        return self.classifier(pooler)

class BERT_Hierarchical_Model(nn.Module):

    def __init__(self, pooling_method="mean", nb_classes=None, model_name=None):
        super(BERT_Hierarchical_Model, self).__init__()

        self.pooling_method = pooling_method

        #self.bert_path = 'bert-base-uncased'
        self.bert = FlaubertModel.from_pretrained(model_name)
        self.pre_classifier = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(256, nb_classes)

    def forward(self, ids, mask, token_type_ids, lengt):
        #print("Inputs ", ids.shape)
        pooled_out, = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids,  return_dict=False)

        # Redimensionner la sortie de Embedding 3D to 2D
        
        pooled_out = pooled_out.permute(0, 2, 1)[:, :, -1]

        chunks_emb = pooled_out.split_with_sizes(lengt)
        
        if self.pooling_method == "mean":
            emb_pool = torch.stack([torch.mean(x, 0) for x in chunks_emb])
        elif self.pooling_method == "max":
            emb_pool = torch.stack([torch.max(x, 0)[0] for x in chunks_emb])

        emb_pool = self.pre_classifier(emb_pool)
        emb_pool = nn.ReLU()(emb_pool)
        emb_pool = self.dropout(emb_pool)
        return self.out(emb_pool)