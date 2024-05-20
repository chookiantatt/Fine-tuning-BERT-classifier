from bert_mulitclass import BERTClass
from trainer import Trainer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import os
import argparse
import joblib
import torch.nn as nn
from transformers import AutoTokenizer
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', required=True)
parser.add_argument('--model_name', required=True)

# script variables
args = parser.parse_args()
## data variables
data_dir = args.data_dir
model_name = args.model_name
## model variable
model_path = '/home/bert_class/models/' + model_name

test_batch_size = 16
LEARNING_RATE = 1e-5
thres = 0.5

#check for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Model device used:", "cuda" if device == "cuda" else "cpu")

testds = load_dataset(data_dir, split='train', cache_dir='/home/doc_class/cache')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='/home/doc_class/cache')

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length',  truncation=True)

encoded = testds.map(preprocess_function, batched=True)
encoded.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

test_dataloader = DataLoader(encoded, batch_size=test_batch_size)

mlb = joblib.load(model_path+'/labels_binarizer')
nclasses = len(mlb.classes_)

model = BERTClass(nclasses)
# model.to(device)
model = nn.DataParallel(model.cuda())

print(f'model initiated with {nclasses} classes')

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trainer = Trainer(None, None, test_dataloader, model, optimizer, model_path=model_path)

print('Predicting')
scores = trainer.predict(device= device)
print('Finish Predicting')

print('Compiling Predictions')
pred_df = pd.DataFrame(scores, columns=mlb.classes_)
pred_df['patent_id'] = encoded['patent_id']
pred_df = pred_df.set_index('patent_id').reset_index(names='patent_id')
pred_df.to_parquet(os.path.splitext(data_dir)[0] + '_' + model_name + '_infer_scores.parquet', index=False)


def get_infer_labs(infers, thres=thres):
    labs = infers[infers >= thres].index.tolist()
    return labs

if thres:
    pred_labs_pd = \
        pred_df.set_index('patent_id').apply(lambda infers: get_infer_labs(infers, thres=thres), axis=1).\
            to_frame('infer_labs').\
            reset_index()

    test_colnames = testds.column_names
    test_true_cols = list(filter(lambda col: 'label' in col, test_colnames))

    for truecol in test_true_cols:
        pred_labs_pd[truecol.replace('label', 'true_labs')] = testds[truecol]
    pred_labs_pd.to_parquet(os.path.splitext(data_dir)[0] + '_' + model_name + '_infer_labs.parquet', index=False)
print('Finish Compilation')