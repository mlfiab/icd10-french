import torch
from torch.utils.data import Dataset
import re
import ast


def tranform_list_string(LIST):
    labels = []
    for label in LIST :
        try:
            labels.append(ast.literal_eval(label))
        except:
            labels.append(label)
        
    return labels

def retrieve_label(dataframe):
    labels = []
    for lab in dataframe.label.values :
        #print(lab)
        #exit()
        labels.append(lab)
        
    return labels

class CustomDocumentDataset(Dataset):
    def __init__(self, tokenizer, max_len, data, max_chunk, chunk_len=200, overlap_len=50, approach="all", max_size_dataset=None, min_len=249, truncated=0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len
        self.approach = approach
        self.min_len = min_len
        self.max_size_dataset = max_size_dataset
        self.truncated = truncated
        self.max_chunk = max_chunk
        #data = data[:100]
        self.data = data
        self.text, self.label = data.text.values, data.label.values
        
        
    def clean_txt(self, text):
        """ Remove special characters from text """

        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text
    
    def long_terms_tokenizer(self, data_tokenize, targets):
        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids = data_tokenize["token_type_ids"].reshape(-1)
        try:
            remain = data_tokenize.get("overflowing_tokens")[0]
        except Exception as e :
            remain = torch.empty(0)
        #print(remain)
    
        targets = torch.tensor(targets, dtype=torch.int)

        input_ids_list.append(previous_input_ids)
        attention_mask_list.append(previous_attention_mask)
        token_type_ids_list.append(previous_token_type_ids)
        targets_list.append(targets)
    
        #print(remain.shape)
        if remain.numel() != 0 and self.approach != 'head':
            remain = torch.tensor(remain, dtype=torch.long)
            idxs = range(len(remain)+self.chunk_len)
            idxs = idxs[(self.chunk_len-self.overlap_len-2)::(self.chunk_len-self.overlap_len-2)]
            input_ids_first_overlap = previous_input_ids[-(
                self.overlap_len+1):-1]
            start_token = torch.tensor([101], dtype=torch.long)
            end_token = torch.tensor([102], dtype=torch.long)

            for i, idx in enumerate(idxs):
                if i == 0:
                    input_ids = torch.cat(
                        (input_ids_first_overlap, remain[:idx]))
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx-self.overlap_len):idx]

                previous_idx = idx

                nb_token = len(input_ids)+2
                attention_mask = torch.ones(self.chunk_len, dtype=torch.long)
                attention_mask[nb_token:self.chunk_len] = 0
                token_type_ids = torch.zeros(self.chunk_len, dtype=torch.long)
                input_ids = torch.cat((start_token, input_ids, end_token))
                if self.chunk_len-nb_token > 0:
                    padding = torch.zeros(
                        self.chunk_len-nb_token, dtype=torch.long)
                    input_ids = torch.cat((input_ids, padding))

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)
                targets_list.append(targets)

        if len(targets_list) < self.max_chunk+1:
            return({
                'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
                # torch.tensor(mask, dtype=torch.long),
                'mask': attention_mask_list,
                # torch.tensor(token_type_ids, dtype=torch.long),
                'token_type_ids': token_type_ids_list,
                'targets': targets_list,
                'len': [torch.tensor(len(targets_list), dtype=torch.long)]
            })
        else:
            return({
                'ids': input_ids_list[:self.max_chunk],  # torch.tensor(ids, dtype=torch.long),
                # torch.tensor(mask, dtype=torch.long),
                'mask': attention_mask_list[:self.max_chunk],
                # torch.tensor(token_type_ids, dtype=torch.long),
                'token_type_ids': token_type_ids_list[:self.max_chunk],
                'targets': targets_list[:self.max_chunk],
                'len': [torch.tensor(len(targets_list[:self.max_chunk]), dtype=torch.long)]
            }) 
    
    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""

        doc_content = str(self.text[idx])
        targets = self.label[idx]
        if self.truncated == 0:
            text = self.tokenizer.encode_plus(
                doc_content,
                max_length=self.chunk_len,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_tensors='pt')
            long_token = self.long_terms_tokenizer(text, targets)
            return long_token
            
        else :
            text = self.tokenizer.encode_plus(
                doc_content,
                max_length=512,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                #return_overflowing_tokens=True,
                return_tensors='pt')
            long_token = self.long_terms_tokenizer(text, targets)
            return long_token

        

    def __getitem__2(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""

        doc_content = str(self.text[idx])
        targets = self.label[idx]
        text = self.tokenizer.encode_plus(
            doc_content,
            max_length=self.chunk_len,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors='pt')
        return text

    def __len__(self):
        """ Return data length """
        return len(self.label)
    
    def __len2__(self):
        return len(self.label[0])