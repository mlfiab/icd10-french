# Automatic ICD-10 code classification system in French
- [Paper](https://doi.ieeecomputersociety.org/10.1109/CBMS58004.2023.00198)

![image](https://github.com/mlfiab/icd10-french/blob/main/global-architecture.png)

## Reference
Please cite the following paper:
```
    @INPROCEEDINGS {10178718,
    author = {Y. Tchouka and J. Couchot and D. Laiymani and P. Selles and A. Rahmani},
    booktitle = {2023 IEEE 36th International Symposium on Computer-Based Medical Systems (CBMS)},
    title = {Automatic ICD-10 Code Association: A Challenging Task on French Clinical Texts},
    year = {2023},
    volume = {},
    issn = {},
    pages = {91-96},
    abstract = {Automatically associating ICD codes with electronic health data is a well-known NLP task in medical research. NLP has evolved significantly in recent years with the emergence of pre-trained language models based on Transformers architecture, mainly in the English language. This paper adapts these models to automatically associate the ICD codes. Several neural network architectures have been experimented with to address the challenges of dealing with a large set of both input tokens and labels to be guessed. In this paper, we propose a model that combines the latest advances in NLP and multi-label classification for ICD-10 code association. Fair experiments on a Clinical dataset in the French language show that our approach increases the $F_{1}$-score metric by more than 55% compared to state-of-the-art results.},
    keywords = {measurement;adaptation models;codes;computational modeling;neural networks;computer architecture;transformers},
    doi = {10.1109/CBMS58004.2023.00198},
    url = {https://doi.ieeecomputersociety.org/10.1109/CBMS58004.2023.00198},
    publisher = {IEEE Computer Society},
    address = {Los Alamitos, CA, USA},
    month = {jun}
    }
```


## Requirements
* Python >= 3.6 (via anaconda recommanded)
* Install the required Python packages with `pip install -r requirements.txt`
* If the specific versions could not be found in your distribution, you could simple remove the version constraint. Our code should work with most versions.

## Dataset
Obviously for privacy reasons, we are not allowed to share the dataset used in this work. For execution you have to put your data in `data` folder

We assume that the dataset contain the columns : 'text' & 'CIM10'
1. Column 'text': Input data containing text medical data 
2. Column 'CIM10': ICD codes list
e.g: ['E86', 'J100', 'E8708', 'J90']

## Architectures
In this work, various architectures has been experimented to tackle the ICD code association challenge with 2 pretrained french models: FlauBERT & CamemBERT
1. Truncated BERT: Finetuning bert models by truncating 512 tokens of the input data
2. Max/Mean pooling: Finetuning bert models with long sequence processing 
3. BERT + LAAT: Finetuning bert models with Label-Aware ATtention mechanism
## How to run

### Training
1. Put the dataset in the folder: `data`
2. Run the following command to train the model.

```
    python main.py --train_file data/DATA.csv
```

### Notes
- If you would like to train the model based on the K most frequent code, set `--most-code <K>` with K the number of the codes to be considered.
