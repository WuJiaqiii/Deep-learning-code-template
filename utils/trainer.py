import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils.utils import AverageMeter, EarlyStopping, save_confusion_matrix
from models.model import *
from models.loss import *

class Trainer:
    def __init__(self, config, data_loader, logger, model):
        
        self.config = config
        self.logger = logger
        self.device = self.config.device

        self.train_loader, self.val_loader = data_loader
        
        if torch.cuda.device_count() > 1 and self.config.use_data_parallel:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for training...")
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)

        if self.config.use_amp_autocast and self.device == 'cuda':
            self.scaler = GradScaler(device="cuda")

        # self.creterion = 
        # self.optimizer = 
        # self.scheduler = 
            
        self.early_stopping = EarlyStopping(logger=self.logger, patience=self.config.early_stop_patience, delta=0)
        self.writer = SummaryWriter(log_dir=config.result_dir)
        
    def train_one_epoch(self, epoch):
        
        loss_record = AverageMeter()
        
        self.model.train()

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)):

            inputs, labels = batch
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            self.optimizer.zero_grad()

            if self.config.use_amp_autocast:
                with autocast():
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            loss_record.update(loss.item(), labels.size(0))
            
        self.logger.info(f'Train Epoch: {epoch + 1}, Avg Loss: {loss_record.avg:.4f} ')
        self.writer.add_scalar("Loss/Train", loss_record.avg, epoch)

    @torch.no_grad() 
    def validate(self, epoch):
        
        loss_record = AverageMeter()

        self.model.eval()

        preds, targets = [], []
        mask = np.array([])
        total_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", leave=True)):

            inputs, labels = batch
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            if self.config.use_amp_autocast:
                with autocast():
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

            loss_record.update(loss.item(), labels.size(0))
            
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(labels.cpu().numpy())   
         
        self.logger.info(f'Validate Epoch: {epoch + 1}, Loss: {loss_record.avg:.4f}')

        preds, targets = np.concatenate(preds), np.concatenate(targets)
        acc = (preds == targets).sum().item() / preds.shape[0]
        self.logger.info(f'Validate Accuracies: {acc:.4f} ')
  
        save_confusion_matrix(targets, preds, self.config, epoch)
        self.writer.add_scalar("Loss/Validation", loss_record.avg, epoch)

        return loss_record.avg

    def train(self):
        
        best_val_loss = np.inf
        for epoch in range(self.config.epochs):
            
            self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            
            self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss 
                best_path = os.path.join(self.config.model_dir, f'model_best.pth')
                if torch.cuda.device_count() > 1 and self.config.use_data_parallel:
                    torch.save(self.model.module.state_dict(), best_path)
                else:
                    torch.save(self.model.state_dict(), best_path)
                
                self.logger.info(f"--Best model saved at epoch {epoch + 1} with loss: {best_val_loss:.4f}")
                
            self.early_stopping(val_loss, self)
            if self.early_stopping.early_stop:
                self.logger.info("--Early stopping triggered")
                break

        if torch.cuda.device_count() > 1 and self.config.use_data_parallel:
            torch.save(self.model.module.state_dict(), os.path.join(self.config.model_dir, f'model_last.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, f'model_last.pth'))
        
        self.writer.close()