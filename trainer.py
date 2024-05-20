import os
import torch
import shutil
import json
from tqdm import tqdm
import numpy as np

class Trainer():
    def __init__(self, training_loader, validation_loader, test_loader, model, optimizer, model_path,
                 logs=[['epoch', 'train_loss', 'val_loss', 'eval_score']]):
        self.train_loader = training_loader
        self.val_loader = validation_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.model_path = model_path
        self.ini_model_dir()
        self.verbose_logs = logs

    def ini_model_dir(self):
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def save_checkpoint(self, current_epoch, eval_score):
        try:
            checkpoint = {'epoch': current_epoch,
                          'eval_score': eval_score,
                          'model_state': self.model.state_dict(),
                          'optimiser_state': self.optimizer.state_dict(),
                          'rng_state': torch.get_rng_state()}
            torch.save(checkpoint, os.path.join(self.model_path, 'latest_checkpoint'))
            with open(f'{self.model_path}/training_logs.json', 'w') as logsout:
                json.dump(self.verbose_logs, logsout)
        except:
            print('saving failed')

    def save_best_model(self):
        latest_cp_path = os.path.join(self.model_path, 'latest_checkpoint')
        best_model_path = os.path.join(self.model_path, 'best_model')
        shutil.copyfile(latest_cp_path, best_model_path)

    def load_model(self):
        if os.path.isdir(self.model_path):
            best_model_path = os.path.join(self.model_path, 'best_model')
            best_model_cp = torch.load(best_model_path)
            self.model.load_state_dict(best_model_cp['model_state'])
        # if model saved as a single file, load directly to model
        else:
            self.model.load_state_dict(torch.load(self.model_path))

    def get_samp_f1_thres(self, pred_logits, ytrue, threshold=0.5):
        epsilon = 1e-7
        pred_probs = torch.sigmoid(pred_logits)
        ypred = (pred_probs > threshold).to(torch.float32)
        ytrue = ytrue.to(torch.float32)
        tp = (ypred * ytrue).sum(axis=1)
        samp_precision = (tp / (ypred.sum(axis=1) + epsilon))
        samp_recall = (tp / (ytrue.sum(axis=1) + epsilon))
        samp_f1 = 2 * (samp_precision * samp_recall) / (samp_precision + samp_recall + epsilon)
        samp_f1_mean = samp_f1.mean()
        return samp_f1_mean.item()

    def resume_training_from_checkpoint(self, device, n_epochs):
        if os.path.exists(os.path.join(self.model_path, 'latest_checkpoint')) and os.path.exists(
                os.path.join(self.model_path, 'best_model')):
            latest_cp_path = os.path.join(self.model_path, 'latest_checkpoint')
            best_model_path = os.path.join(self.model_path, 'best_model')
            best_model_cp = torch.load(best_model_path)
            best_eval_score = best_model_cp['eval_score']
            latest_model_cp = torch.load(latest_cp_path)
            last_epoch = latest_model_cp['epoch']
            last_rng_state = latest_model_cp['rng_state']
            self.model.load_state_dict(latest_model_cp['model_state'])
            self.optimizer.load_state_dict(latest_model_cp['optimiser_state'])
            print(f'Resuming training from epoch {last_epoch}')
            with open(f'{self.model_path}/training_logs.json', 'r') as logsin:
                self.verbose_logs = json.load(logsin)
            torch.set_rng_state(last_rng_state)
            self.train(device=device, start_epoch=last_epoch + 1, n_epochs=n_epochs, best_valid_eval=best_eval_score)
        else:
            print('checkpoint does not exist')

    def train(self, device, start_epoch=0, n_epochs=2, best_valid_eval=0.0):
        for epoch in range(start_epoch, n_epochs):
            train_loss = 0
            valid_loss = 0

            self.model.train()
            for batch_idx, data in enumerate(tqdm(self.train_loader, desc=f'epoch {epoch}', position=0, leave=False)):
                # print('yyy epoch', batch_idx)
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                # if batch_idx%5000==0:
                #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print('before loss data in training', loss.item(), train_loss)
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
                # print('after loss data in training', loss.item(), train_loss)

            ######################
            # validate the model #
            ######################

            total_val_samp_f1 = 0
            self.model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(self.val_loader, desc='validation', position=1, leave=False), 0):
                    # ids = data['input_ids'].to(device, dtype=torch.long)
                    # mask = data['attention_mask'].to(device, dtype=torch.long)
                    # token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    # targets = data['target'].to(device, dtype=torch.float)
                    ids = data['input_ids'].to(dtype=torch.long).cuda()
                    mask = data['attention_mask'].to( dtype=torch.long).cuda()
                    token_type_ids = data['token_type_ids'].to(dtype=torch.long).cuda()
                    targets = data['target'].to(dtype=torch.float).cuda()
                    outputs = self.model(ids, mask, token_type_ids)

                    loss = self.loss_fn(outputs, targets)
                    valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                    batch_samp_avg = self.get_samp_f1_thres(outputs, targets)
                    # bug fix here len(data) will return number of fields
                    total_val_samp_f1 += (batch_samp_avg * len(data['input_ids']))

            # calculate average losses
            # print('before cal avg train loss', train_loss)
            train_loss = train_loss / len(self.train_loader)
            valid_loss = valid_loss / len(self.val_loader)
            val_samp_f1 = total_val_samp_f1 / len(self.val_loader.dataset)
            # print training/validation statistics
            print(
                'Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f} \tF1 sample: {:.6f}'.format(
                    epoch, train_loss, valid_loss, val_samp_f1))

            self.verbose_logs.append([epoch, train_loss, valid_loss, val_samp_f1])

            # save checkpoint
            self.save_checkpoint(epoch, val_samp_f1)

            if val_samp_f1 > best_valid_eval:
                self.save_best_model()
                best_valid_eval = val_samp_f1

    def predict_step(self, data, device, k):
        self.model.eval()
        with torch.no_grad():
            # change have
            # scores, labels = torch.topk(self.model(data_x), k)
            # return torch.sigmoid(scores).cpu(), labels.cpu()
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            scores = self.model(ids, mask, token_type_ids)
            if k:
                scores = torch.topk(scores, k)
            return torch.sigmoid(scores).cpu()

    def predict(self, device, k=None, desc='Predict', **kwargs):
        self.load_model()
        # change here
        # scores_list, labels_list = zip(*(self.predict_step(data_x, k)
        # 								 for data_x in tqdm(data_loader, desc=desc, leave=False)))
        # return np.concatenate(scores_list), np.concatenate(labels_list)
        scores_list = [self.predict_step(data_x, device, k=k) for data_x in tqdm(self.test_loader, desc=desc, leave=False)]
        return np.concatenate(scores_list)