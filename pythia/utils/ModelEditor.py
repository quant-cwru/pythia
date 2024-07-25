import torch.nn as nn
from typing import Dict, Any

class ModelEditor:
    """
    This class manages multiple models, their hyperparameters, and provides functionality to edit and view model structures.

    Attributes:
        models (Dict[str, nn.Module]): Dictionary storing model names and their corresponding PyTorch modules.
        hyperparam_configs (Dict[str, Dict[str, Any]]): Dictionary storing hyperparameter configurations for each model.
    """

    def __init__(self):
        """
        Initialize the ModelEditor with empty dictionaries for models and hyperparameter configurations.
        """
        self.models = {}
        self.hyperparam_configs = {}

    def add_model(self, name: str, model: nn.Module):
        """
        Add a new model to the editor.

        Args:
            name (str): The name of the model to be added.
            model (nn.Module): The PyTorch model to be added.
        """
        self.models[name] = model
        self.hyperparam_configs[name] = {}

    def get_model(self, name: str) -> nn.Module:
        """
        Retrieve a model by name.

        Args:
            name (str): The name of the model to retrieve.

        Returns:
            nn.Module: The requested PyTorch model, or None if not found.
        """
        return self.models.get(name)

    def list_models(self):
        """
        List all available models.

        Returns:
            list: A list of all model names currently stored in the editor.
        """
        return list(self.models.keys())

    def set_hyperparams(self, model_name: str, hyperparams: Dict[str, Any]):
        """
        Set hyperparameters for a specific model.

        Args:
            model_name (str): The name of the model to set hyperparameters for.
            hyperparams (Dict[str, Any]): A dictionary of hyperparameter names and their values.

        Raises:
            ValueError: If the specified model_name is not found in the editor.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        self.hyperparam_configs[model_name] = hyperparams

    def get_hyperparams(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific model.

        Args:
            model_name (str): The name of the model to retrieve hyperparameters for.

        Returns:
            Dict[str, Any]: A dictionary of hyperparameter names and their values for the specified model.
                            Returns an empty dictionary if the model or its hyperparameters are not found.
        """
        return self.hyperparam_configs.get(model_name, {})

    def edit_model(self, model_name: str, new_layers: Dict[str, nn.Module]):
        """
        Edit a model by replacing or adding layers.

        Args:
            model_name (str): The name of the model to edit.
            new_layers (Dict[str, nn.Module]): A dictionary of layer names and their corresponding new PyTorch modules.

        Raises:
            ValueError: If the specified model_name is not found in the editor.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        for name, layer in new_layers.items():
            if hasattr(model, name):
                setattr(model, name, layer)
            else:
                model.add_module(name, layer)

    def view_model(self, model_name: str):
        """
        View the structure of a specific model and its hyperparameters.

        Args:
            model_name (str): The name of the model to view.

        Raises:
            ValueError: If the specified model_name is not found in the editor.

        Prints:
            - The name of the model
            - The structure of the model
            - The hyperparameters for the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        print(f"Model: {model_name}")
        print(model)
        print("\nHyperparameters:")
        print(self.get_hyperparams(model_name))