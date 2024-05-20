from datasets import Dataset, load_dataset, Features, Value, ClassLabel, DatasetDict
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
import os
import joblib
import json
from logzero import logger

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_dir', required=True)
parser.add_argument('--valdataset_dir', default='')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--model_dir', required=True)
parser.add_argument('--tokenizer_dir', required=True)

# script variables
args = parser.parse_args()
## data variables
dataset_dir = args.dataset_dir
valdataset_dir = args.valdataset_dir
output_dir = args.output_dir
model_dir = args.model_dir
tokenizer_dir = args.tokenizer_dir
val_size = 0.2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

#check for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Model device used:", "cuda" if device == "cuda" else "cpu")


if len(valdataset_dir) == 0:
    try:
        train_dataset = load_dataset(dataset_dir,
                                    split='train')
        print(f'data loaded from {dataset_dir} of rows {train_dataset.num_rows}')
    
    except Exception as e:
        print(f"Error loading data: {e}")

    dataset_split = train_dataset.train_test_split(test_size=val_size)   # Split to train and validation first
    validation_test_dataset = dataset_split["test"].train_test_split(test_size=0.5)  # Test set = split half from validation set

    train_dataset = dataset_split["train"]
    validation_dataset = validation_test_dataset["train"]
    test_dataset = validation_test_dataset["test"]

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation": validation_dataset
    })

    print(dataset_dict)

else:
    train_dataset = load_dataset(dataset_dir,
                                 split='train',
                                 cache_dir='/home/subs_class/train/cache'
                                 )
    print(f'train data loaded from {dataset_dir} of rows {train_dataset.num_rows}')
    val_dataset = load_dataset(valdataset_dir,
                               split='train',
                               cache_dir='/home/subs_class/val/cache'
                               )
    print(f'val data loaded from {valdataset_dir} of rows {val_dataset.num_rows}')

    ## Store train and test dataset in dictionary
    dataset_split = DatasetDict({'train': train_dataset,
                                 'test': val_dataset})


mlb = MultiLabelBinarizer(sparse_output=True) ## Set to True if output binary array is desired in CSR sparse format
# mlb.fit(dataset_split['train']['label'])
logger.info('Fitting label binarizer from train data')
targets = mlb.fit_transform(dataset_split['train']['label'])  ## transform to binary representation
logger.info('Binarize validation labels')
val_targets = mlb.transform(dataset_split['test']['label'])  ## Transform the given label sets

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
joblib.dump(mlb, os.path.join(model_dir, 'labels_binarizer'))

logger.info('Adding binarised labels to train')
dataset_split['train'] = dataset_split['train'].add_column('target', targets.toarray().tolist())
logger.info('Adding binarised labels to validation')
dataset_split['test'] = dataset_split['test'].add_column('target', val_targets.toarray().tolist())

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='/home/subs_class/tokenizer/cache')

# Add new vocab into tokens
with open("subunits.json", "r") as json_file:
    new_tokens = json.load(json_file)

added_tokens = tokenizer.add_tokens(new_tokens)

tokenizer.save_pretrained(tokenizer_dir)

# tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, cache_dir='/home/subs_class/tokenizer/cache')

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length',  truncation=True)

def binarize_labels(rows):
    target = mlb.transform(rows['label']).toarray().tolist()
    return {'target':target}

logger.info('Tokenising texts')
encoded = dataset_split.map(preprocess_function, batched=True)
encoded = encoded.map(binarize_labels, batched=True)

encoded.save_to_disk(output_dir)
