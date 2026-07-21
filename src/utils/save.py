import os
import joblib
from pathlib import Path

def save_model(model, model_name: str, version: str):
    """
    Save a model to the 'models/' folder using dynamic path based on notebook location.
    
    Format file: {model_name}.v{version}.pkl
    
    Args:
        model: Model object to save (scaler, sklearn model, xgb model, etc.)
        model_name (str): Name of the model, e.g., "scaler", "xgb_model"
        version (str): Version identifier, e.g., "1", "1.0", "2.3"
    
    Returns:
        str: Full path to saved file
    """

    # Get current notebook directory dynamically
    base_path = Path(os.getcwd())        # directory tempat notebook dijalankan
    models_dir = base_path / "models" / f"v{version}"
    models_dir.mkdir(parents=True, exist_ok=True)

    # File name format
    file_name = f"{model_name}.pkl"
    full_path = models_dir / file_name

    # Save model
    joblib.dump(model, full_path)

    print(f"✅ Saved: {full_path}")
    return str(full_path)

def save_features(feature_list, version: str):
    """
    Save a list of features used by the model.
    File name: features.pkl
    Stored in: models/v{version}/
    """
    base_path = Path(os.getcwd())
    models_dir = base_path / "models" / f"v{version}"
    models_dir.mkdir(parents=True, exist_ok=True)

    full_path = models_dir / "features.pkl"
    joblib.dump(feature_list, full_path)

    print(f"📄 Features saved: {full_path}")
    return str(full_path)