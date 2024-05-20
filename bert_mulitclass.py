import torch
from transformers import AutoModel

class BERTClass(torch.nn.Module):
    def __init__(self, num_labels, tokenizer):
        super(BERTClass, self).__init__()
        self.num_labels = num_labels
        self.l1 = AutoModel.from_pretrained('bert-base-uncased', cache_dir='/home/subs_class/model/cache')
        self.tokenizer = tokenizer
        self.l1.resize_token_embeddings(len(tokenizer))  ## Random embeddings for new token
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, self.num_labels)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids).pooler_output
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output