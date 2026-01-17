import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, device= 'cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        
        #loss function (weighted for class imbalance)
        self.criterion = nn.CrossEntropyLoss()
        
        #optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr= cfg['training']['learning_rate'],
            weight_decay= cfg['training']['weight_decay']
        )
        
        #learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        #tracking 
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        #paths
        self.checkpoint_dir = Path(cfg['training']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
       
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            #forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            #backward
            loss.backward()
            self.optimizer.step()
            
            #metrics
            running_loss+= loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            #update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
                
            })    
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.* correct / total
        return epoch_loss, epoch_acc
    
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                #forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                #metrics
                running_loss+= loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                #update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                    
                })     
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.* correct / total
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, val_loss, is_best = False):
        checkpoint= {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'historay': self.history
        }
        
        #save latest
        path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, path)
        
        #save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"    Best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")
            
    def train(self, num_epochs):
        print(f"\n{'='*50}")
        print(f" Starting Training for {num_epochs} epochs...")
        print(f"{'='*50}\n")
        
        for epoch in range(1, num_epochs +1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            print("-"*50)
            
            #train 
            train_loss, train_acc = self.train_epoch()
            
            #validate
            val_loss, val_acc = self.validate()
            
            #update scheduler
            self.scheduler.step(val_loss)
            
            #save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            #print summary
            print(f"\n Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
            
            #save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
                
                    