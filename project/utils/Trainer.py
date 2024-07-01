import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model_name, data_processor, model_editor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.model = model_editor.get_model(model_name)
        self.data_processor = data_processor
        self.model_editor = model_editor
        self.device = device
        self.model.to(self.device)
        
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        self.train_losses = []
        self.val_losses = []
        
        # Get hyperparameters from model editor
        self.hyperparams = self.model_editor.get_hyperparams(model_name)
        
    def setup_training(self):
        # Set up criterion, optimizer, and scheduler based on hyperparameters
        loss_function = self.hyperparams.get('loss_function', 'mse')
        if loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_function == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        optimizer_name = self.hyperparams.get('optimizer', 'adam')
        lr = self.hyperparams.get('learning_rate', 0.001)
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        scheduler_name = self.hyperparams.get('scheduler', None)
        if scheduler_name == 'step_lr':
            step_size = self.hyperparams.get('scheduler_step_size', 10)
            gamma = self.hyperparams.get('scheduler_gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name is not None:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
    def train(self):
        num_epochs = self.hyperparams.get('num_epochs', 50)
        batch_size = self.hyperparams.get('batch_size', 32)
        validation_split = self.hyperparams.get('validation_split', 0.2)
        snapshot_interval = self.hyperparams.get('snapshot_interval', 5)
        
        dataset = self.data_processor.to_torch()
        train_size = int((1 - validation_split) * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.scheduler:
                self.scheduler.step()
            
            if (epoch + 1) % snapshot_interval == 0:
                self.save_snapshot(f"snapshot_{self.model_name}_epoch_{epoch+1}.pth")
        
        self.plot_loss()
    
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} - Training and Validation Loss')
        plt.legend()
        plt.savefig(f'{self.model_name}_loss_plot.png')
        plt.close()
    
    def save_snapshot(self, filename):
        os.makedirs('snapshots', exist_ok=True)
        path = os.path.join('snapshots', filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'hyperparams': self.hyperparams
        }, path)
        print(f"Snapshot saved to {path}")
    
    def load_snapshot(self, filename):
        path = os.path.join('snapshots', filename)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.hyperparams = checkpoint['hyperparams']
        print(f"Snapshot loaded from {path}")