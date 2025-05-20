import torch


def save_model(model, optimizer, save_path):
    """
    Save the model and optimizer state to a specified path.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        save_path (str): Path to save the model file.
    """
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)
        print(f"Model saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model_state(model, optimizer, model_path):
    """
    Load the model state from a specified path.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        model_path (str): Path to the saved model file.
        
    Returns:
        torch.nn.Module: The model with loaded state.
    """
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model state loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model state: {e}")
    
    return model, optimizer