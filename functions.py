import torch
import pandas as pd
import numpy as np

import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sparse
from sklearn import metrics
from sklearn.metrics import classification_report

import ast
import time

from collections import Counter
from losses import AsymmetricLoss

def asymetric_loss(outputs, targets):
    loss = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    return loss(outputs, targets)

def evaluate(targets, predicted, save_score=None):
    THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    RESULTS = []

    for threshold in THRESHOLDS:
        outputs = (np.array(predicted) >= threshold).astype(int)
        
        precision = round(metrics.precision_score(np.array(targets), np.array(outputs), average='micro'),2)
        recall = round(metrics.recall_score(targets, np.array(outputs), average='micro'),2)
        fscore = round(metrics.f1_score(targets, np.array(outputs), average='micro'),2)

        RESULTS.append((precision, recall, fscore))
        #print(metrics.multilabel_confusion_matrix(np.array(targets), np.array(outputs)))
        #print(classification_report(targets, outputs))
        if save_score:
            with open(save_score, 'a') as score :
                print(classification_report(targets, outputs), file=score)
        #accuracy = metrics.accuracy_score(targets, np.array(outputs))
        
    return{
        "results": RESULTS,
        "nb exemple": len(targets)}
    
def evaluate_(target, predicted):
    true_label_mask = [1 if (np.argmax(x)-target[i]) == 0 else 0 for i, x in enumerate(predicted)]
    nb_prediction = len(true_label_mask)
    true_prediction = sum(true_label_mask)
    false_prediction = nb_prediction-true_prediction
    accuracy = true_prediction/nb_prediction
    return{
        "accuracy": accuracy,
        "nb exemple": len(target),
        "true_prediction": true_prediction,
        "false_prediction": false_prediction,
    }
    
def rnn_train_loop_fun1(data_loader, model, optimizer, device, loss_, scheduler=None):
    model.train()
    t0 = time.time()
    losses = []

    for batch_idx, batch in enumerate(data_loader):

        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"][0] for data in batch]
        lengt = [data['len'] for data in batch]
        
        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.stack(targets)
        lengt = torch.cat(lengt)
        lengt = [x.item() for x in lengt]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        optimizer.zero_grad()
        #print(ids.shape, lengt)
        try:
            outputs = model(ids=ids, mask=mask,
                            token_type_ids=token_type_ids, lengt=lengt)
            
            loss = loss_(outputs, targets)
            loss.backward()
            model.float()
            optimizer.step()
        
            if scheduler:
                scheduler.step()
            losses.append(loss.item())
            #del loss
            #del outputs
        except Exception as e :
            print("Error Memory : ", ids.shape, lengt)
            print(e)

        if batch_idx % 5 == 0:
            print(f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()

    return losses

def rnn_eval_loop_fun1(data_loader, model, device, loss_):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):

            #model.half()
            ids = [data["ids"] for data in batch]
            mask = [data["mask"] for data in batch]
            token_type_ids = [data["token_type_ids"] for data in batch]
            targets = [data["targets"][0] for data in batch]
            lengt = [data['len'] for data in batch]

            ids = torch.cat(ids)
            mask = torch.cat(mask)
            token_type_ids = torch.cat(token_type_ids)
            targets = torch.stack(targets)
            lengt = torch.cat(lengt)
            lengt = [x.item() for x in lengt]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            try:
                outputs = model(ids=ids, mask=mask,token_type_ids=token_type_ids, lengt=lengt)
                loss = loss_(outputs, targets)
                losses.append(loss.item())
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            except Exception as e :
                print("Error Memory : ", ids.shape, lengt)

    return fin_outputs, fin_targets, losses

def transform_label(unique_icd_10_list, dataset_icd_10_labels):
    mlb = MultiLabelBinarizer()
    mlb.fit([unique_icd_10_list])
    nb_classes = len(list(mlb.classes_))
    print("Nombre de classes ==> LABELS : ", nb_classes)

    dataset_icd_10_labels = tranform_list_string(dataset_icd_10_labels)
    icds = mlb.transform(dataset_icd_10_labels)
    len_data = len(dataset_icd_10_labels)
    arr = sparse.coo_matrix((icds), shape = (len_data, nb_classes))
    return arr.toarray().tolist()

def group_cim_10_family(df):
    codes = list(df['CIM-10'])
    codes = [x for x in codes if str(x) != 'nan']
    codes = [cim[:3] for cim in codes]
    codes = list(dict.fromkeys(codes))

    cim10_family = pd.DataFrame({'CIM10 Family':codes})
    cim10_family.to_csv('cim10_family.csv', sep=";", index=False)
    
    return cim10_family

def text_mining(df):

    df['len_txt'] = df['text'].apply(lambda x : len(x.split()))
    return df

def retrieve_k_common_icd(dataframe, K, strategy='most'):
    CODES_LABELS = list(dataframe["CIM10"])
    CODES_LABELS = tranform_list_string(CODES_LABELS)
    CODES = retrieve_all_codes(CODES_LABELS)
    codes_count = Counter(CODES)
    #codes_frequency = codes_count.most_common()
    #breakpoint()
    #codes_freq = pd.DataFrame(codes_frequency, columns=["Codes", "Frequency"])
    #codes_freq.to_csv("code_freq.csv", sep=";", index=False)
    if strategy == "most":
        common_codes = codes_count.most_common(K)
    else :
        common_codes = codes_count.most_common()[-K:]
    unique_icd_10_list = []
    for cim in common_codes :
        code, occ = cim
        unique_icd_10_list.append(code)

    return unique_icd_10_list

def retrieve_all_codes(codes_labels):
    CODES = []
    for codes in codes_labels :
        for code in codes:
            CODES.append(code)

    return CODES

def deleted_status(label):
    try:
        label = ast.literal_eval(label)
    except:
        label = label
    if label.count(1) == 0:
        return True
    else:
        return False

def del_no_label_row(df):
    df['Deleted'] = df['label'].apply(lambda x : deleted_status(x))
    #df = df.loc[df['Deleted'] == False]
    #del df['Deleted']
    return df

def replace_no_common_tag(dataset_icd_10_labels, unique_code_list):
    dataset_icd_10_labels = tranform_list_string(dataset_icd_10_labels)
    dataset_icd_10_labels_new = []
    for label in dataset_icd_10_labels:
        new_label = []
        for c in label : 
            if c in unique_code_list :
                new_label.append(c)
            else:
                new_label.append("EXTRA")
        new_label = list(dict.fromkeys(new_label))
        dataset_icd_10_labels_new.append(new_label)
        unique_code_list.append("EXTRA")
    return dataset_icd_10_labels_new, unique_code_list 

def add_extra_label(df):
    #labels = retrieve_label(df)
    labels_with_extra_code =  []
    for index, row in df.iterrows():
        label = row['label']
        status = row['Deleted']
        #print(label, status)
        if status:
            label.append(1)
        else:
            label.append(0)
        labels_with_extra_code.append(label)

    del df['label']
    df['label'] = labels_with_extra_code
    return df

def my_collate1(batches):
    # return batches
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]

def loss_fun(outputs, targets):
    #targets = targets.type_as(outputs)
    loss = nn.BCEWithLogitsLoss()
    #print("Erreur",outputs.shape, targets.shape )
    return loss(outputs, targets)
    # return nn.BCEWithLogitsLoss()(outputs, targets)

def loss_fun_2(outputs, targets):
    loss = nn.MultiLabelSoftMarginLoss()

    #Apply Sigmoid activation on outputs
    outputs = torch.sigmoid(outputs)
    return loss(outputs, targets)

def asymetric_loss(outputs, targets):
    loss = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    return loss(outputs, targets)

def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))
        return device

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        return device
    
def retrieve_classes(df):
    codes = list(df['CIM10'])
    codes = list(dict.fromkeys(codes))
    return codes

if __name__ == '__main__':
    pass

def tranform_list_string(LIST):
    labels = []
    for label in LIST :
        try:
            labels.append(ast.literal_eval(label))
        except:
            labels.append(label)
        
    return labels