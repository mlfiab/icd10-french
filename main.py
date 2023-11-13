import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AdamW # get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
from transformers import get_linear_schedule_with_warmup
import argparse
import time
import sys

from functions import *
from models import *
from dataset import *

def read_data(path_file) : 
    df = pd.read_csv(path_file, sep=';')
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
  
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv file containing the training data."
    )

    parser.add_argument(
        "--most_code",
        type=int,
        help="All codes or the most frequent system",
        default=None, 
    )
    
    parser.add_argument(
    "--model_type",
    type=str,
    help="The type of model",
    default="camem-laat",
    choices=["laat", "camem-laat", "hier_mean", "hier_mean", "trunc-bert"]
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # label_task = int(sys.argv[1])
    # loss_strategy = str(sys.argv[2])
    #ARCHITECTURE = Flaubert or Camembert
    # lr = float(sys.argv[3])
    #K = 0
    args = parse_args()
    #path_file = 'data/DATA.csv' # we suppose DATA.csv file in data folder
    #Dataset HNFC preprocessing -----
    data = read_data(args.train_file)
    classes = retrieve_classes(data)
    unique_icd_10_list = classes
    
    #classes = classes[~classes.isnull().any(axis=1)]
    dataset_icd_10_labels = list(data["CIM10"])
    # classes_in_family = group_cim_10_family(classes)
    try:
        # K Top/Last common ICD-10
        K = args.mode
        print(K)
        unique_icd_10_list = retrieve_k_common_icd(data, K, strategy='most')
        # Transform codes matrix binary classification

        # Replace all extra codes by other tag
        dataset_icd_10_labels, unique_icd_10_list = replace_no_common_tag(dataset_icd_10_labels, unique_icd_10_list)
    except Exception as e:
        print(e)
        pass
    labels = transform_label(unique_icd_10_list, dataset_icd_10_labels)
    nb_classes = len(labels[1])
    print("LABELS : => ", nb_classes)
    
    data["label"] = labels

    # # Get length token in sequence
    data = text_mining(data)

    data = data[data['text'].notnull()]

    TRAIN_BATCH_SIZE=4
    EPOCH=30 #[5,10,20,30]
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    MIN_LEN=200
    MAX_LEN = 2000
    CHUNK_LEN=200
    OVERLAP_LEN=50
    #MAX_LEN=10000000
    #MAX_SIZE_DATASET=1000
    device = get_device()
    #device = 'cpu'
    print("Device : ", device)
    
    # MODEL_NAME = 'camembert-base joeddav/xlm-roberta-large-xnli' #'flaubert/flaubert_base_cased flaubert/flaubert_small_cased'
        

    #architectures = ['camembert-laat'] # [hierarchical_mean/max, laat, truncated-bert, ]
    model = None
    trunc = 0
    MAX_CHUNK = 30
    MODEL_NAME = ""

    arch = args.model_type
    if arch == 'camem-laat':
        MODEL_NAME = 'camembert-base'
        hidden = 768
        model = BERT_Hierarchical_Model_Laat(nb_classes, MODEL_NAME, hidden).to(device)
    if arch == 'laat': #FlauBERT
        hidden = 512
        MODEL_NAME = 'flaubert/flaubert_small_cased'
        model = BERT_Hierarchical_Model_Laat(nb_classes, MODEL_NAME, hidden).to(device)
    if len(arch.split('_')) == 2:
        MODEL_NAME = 'flaubert/flaubert_small_cased'
        _, pooling_method = arch.split('_')
        model=BERT_Hierarchical_Model(pooling_method, nb_classes, MODEL_NAME).to(device)
    if arch == "trunc-bert":
        trunc = 1
        MODEL_NAME = 'flaubert/flaubert_small_cased'
        model = FlauBERTClass(nb_classes, MODEL_NAME).to(device)

    print('Loading BERT tokenizer...')
    bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    # DATASET
    dataset=CustomDocumentDataset(
    tokenizer=bert_tokenizer,
    data=data,
    min_len=MIN_LEN,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    truncated = trunc,
    #max_size_dataset=MAX_SIZE_DATASET,
    overlap_len=OVERLAP_LEN,
    max_chunk = MAX_CHUNK)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    print(dataset_size)
    text = dataset.__getitem__(0)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_data_loader=DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=my_collate1,
        num_workers=2)

    valid_data_loader=DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=valid_sampler,
        collate_fn=my_collate1,
        num_workers=2)

    lr=3e-5 #1e-3##
    #device = 'cpu'
    num_training_steps=int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

    
    optimizer=AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = num_training_steps)
    val_losses=[]
    batches_losses=[]
    val_acc=[]
    print(f"Architecture : {arch} Learning Rate {lr} Batch size {TRAIN_BATCH_SIZE}")
    save_score = None

    for epoch in range(EPOCH):
        t0 = time.time()    
        print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")

        batches_losses_tmp=rnn_train_loop_fun1(train_data_loader, model, optimizer, device, loss_=loss_fun)
        epoch_loss=np.mean(batches_losses_tmp)
    
        print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
        t1=time.time()
        #val_acc.append(tmp_evaluate['accuracy'])
        output, target, val_losses_tmp=rnn_eval_loop_fun1(valid_data_loader, model, device, loss_=loss_fun)
        val_losses.append(val_losses_tmp)
        batches_losses.append(batches_losses_tmp)

    
        print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
        #save_score = f"model__epoch{epoch+1}_maxlen{MAX_LEN}_lr{lr}_batch{TRAIN_BATCH_SIZE}_.txt" 
        tmp_evaluate=evaluate(target, output, save_score)
        print(f"=====>\t{tmp_evaluate}")
    print("\t§§ the Model has been saved §§")
    torch.save(model, f"model_maxlen{MAX_LEN}_lr{lr}_batch{TRAIN_BATCH_SIZE}_.pt")
