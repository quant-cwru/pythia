import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class Trainer:
    """
    This class manages the training process for a PyTorch model, including setup, training loop, loss tracking, snapshot saving/loading, and loss visualization.

    Attributes:
        model_name (str): Name of the model being trained.
        model (nn.Module): The PyTorch model to be trained.
        data_processor (DataProcessor): Object handling data processing.
        model_editor (ModelEditor): Object managing model configurations.
        device (str): Device to run the training on ('cuda' or 'cpu').
        criterion (nn.Module): Loss function for training.
        optimizer (optim.Optimizer): Optimization algorithm.
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_losses (list): List to store training losses.
        val_losses (list): List to store validation losses.
        hyperparams (dict): Dictionary of hyperparameters for training.
    """

    def __init__(self, model_name, data_processor, model_editor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Trainer with a model and necessary components.

        Args:
            model_name (str): Name of the model to train.
            data_processor (DataProcessor): Object for data processing.
            model_editor (ModelEditor): Object for model management.
            device (str): Device to run the training on. Defaults to cuda if available, else cpu.
        """
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
        self.is_lstm = isinstance(self.model, nn.LSTM)
        
    def setup_training(self):
        """
        Set up the training components based on the hyperparameters.
        This includes the loss function, optimizer, and learning rate scheduler.

        Raises:
            ValueError: If an unsupported loss function, optimizer, or scheduler is specified.
        """
        # Set up loss function
        loss_function = self.hyperparams.get('loss_function', 'mse')
        if loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_function == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # Set up optimizer
        optimizer_name = self.hyperparams.get('optimizer', 'adam')
        lr = self.hyperparams.get('learning_rate', 0.001)
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Set up learning rate scheduler
        scheduler_name = self.hyperparams.get('scheduler', None)
        if scheduler_name == 'step_lr':
            step_size = self.hyperparams.get('scheduler_step_size', 10)
            gamma = self.hyperparams.get('scheduler_gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name is not None:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        # Set up lstm
        if self.is_lstm:
            self.sequence_length = self.hyperparams.get('sequence_length', 10)
            self.data_processor.sequence_length = self.sequence_length 
    def train(self):
        """
        Execute the training loop for the model.

        This method handles the entire training process, including:
        - Splitting the data into training and validation sets
        - Running the training loop for the specified number of epochs
        - Computing and storing training and validation losses
        - Updating the learning rate if a scheduler is used
        - Saving model snapshots at specified intervals
        - Plotting the loss curves after training
        """
        # Extract hyperparameters
        num_epochs = self.hyperparams.get('num_epochs', 50)
        batch_size = self.hyperparams.get('batch_size', 32)
        validation_split = self.hyperparams.get('validation_split', 0.2)
        snapshot_interval = self.hyperparams.get('snapshot_interval', 5)
        
        # Prepare datasets and dataloaders
        dataset = self.data_processor.to_torch(is_lstm=self.is_lstm)    
        train_size = int((1 - validation_split) * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
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
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets.squeeze())
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Update learning rate if scheduler is used
            if self.scheduler:
                self.scheduler.step()
            
            # Save snapshot at specified intervals
            if (epoch + 1) % snapshot_interval == 0:
                self.save_snapshot(f"snapshot_{self.model_name}_epoch_{epoch+1}.pth")
        
        # Plot loss curves after training
        self.plot_loss()
    
    def plot_loss(self):
        """
        Plot and save the training and validation loss curves as '{model_name}_loss_plot.png'.
        """
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
        """
        Save a snapshot of the current model state.

        Args:
            filename (str): Name of the file to save the snapshot.
        """
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
        """
        Load a previously saved snapshot.

        Args:
            filename (str): Name of the file to load the snapshot from.
        """
        path = os.path.join('snapshots', filename)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.hyperparams = checkpoint['hyperparams']
        print(f"Snapshot loaded from {path}")