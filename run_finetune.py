from bert_mulitclass import BERTClass
from trainer import Trainer
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
import torch
import os
import argparse
import joblib
import torch.nn as nn


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datasetdict_dir', required=True)
parser.add_argument('--checkpoint_dir', default='')
parser.add_argument('--tokenizer_dir', required=True)
parser.add_argument('--model_name', required=True)

# script variables
args = parser.parse_args()
## data variables
datasetdict_dir = args.datasetdict_dir
checkpoint_dir = args.checkpoint_dir
tokenizer_dir = args.tokenizer_dir
model_name = args.model_name

#bert_class_randomwordcount_allppi_1m_app3
## model variable
model_path = '/home/subs_class/models/' + model_name

train_batch_size = 8
eval_batch_size = 8
epoch = 8
LEARNING_RATE = 1e-5

#check for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Model device used:", "cuda" if device == "cuda" else "cpu")

encoded = load_from_disk(datasetdict_dir)
encoded.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'target'])

train_dataloader = DataLoader(encoded['train'], batch_size=train_batch_size)
val_dataloader = DataLoader(encoded['test'], batch_size=eval_batch_size)

if os.path.exists(model_path+'/labels_binarizer'):
    mlb = joblib.load(model_path+'/labels_binarizer')
    nclasses = len(mlb.classes_)
else:
    nclasses = len(encoded['train']['target'][0])


tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, cache_dir='/home/subs_class/tokenizer/cache')

model = BERTClass(nclasses, tokenizer)
# model.to(device)
model = nn.DataParallel(model.cuda())

print(f'model initiated with {nclasses} classes')

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trainer = Trainer(train_dataloader, val_dataloader, val_dataloader, model, optimizer, model_path=model_path)

print(f'Training')
trainer.train(device= device, start_epoch=0,  n_epochs=epoch)

# print(f'Resume Training')
# trainer.resume_training_from_checkpoint(device= device, n_epochs=epoch)