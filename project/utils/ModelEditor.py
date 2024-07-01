import torch.nn as nn
from typing import Dict, Any

class ModelEditor:
    def __init__(self):
        self.models = {}
        self.hyperparam_configs = {}

    def add_model(self, name: str, model: nn.Module):
        """Add a new model to the editor."""
        self.models[name] = model
        self.hyperparam_configs[name] = {}

    def get_model(self, name: str) -> nn.Module:
        """Retrieve a model by name."""
        return self.models.get(name)

    def list_models(self):
        """List all available models."""
        return list(self.models.keys())

    def set_hyperparams(self, model_name: str, hyperparams: Dict[str, Any]):
        """Set hyperparameters for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        self.hyperparam_configs[model_name] = hyperparams

    def get_hyperparams(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters for a specific model."""
        return self.hyperparam_configs.get(model_name, {})

    def edit_model(self, model_name: str, new_layers: Dict[str, nn.Module]):
        """Edit a model by replacing or adding layers."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        for name, layer in new_layers.items():
            if hasattr(model, name):
                setattr(model, name, layer)
            else:
                model.add_module(name, layer)

    def view_model(self, model_name: str):
        """View the structure of a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        print(f"Model: {model_name}")
        print(model)
        print("\nHyperparameters:")
        print(self.get_hyperparams(model_name))