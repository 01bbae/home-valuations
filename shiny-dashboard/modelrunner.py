import xgboost as xgb
import sklearn
from typing import Tuple, List, Optional
import os
import json
import pandas as pd


def load_xgboost_model(
        save_path: str,
        model_name: str = "model"
    ) -> Tuple[xgb.XGBRegressor, List[str]]:
        """
        Load XGBoost model and its feature names from disk.
        
        Args:
            save_path: Directory containing the model files
            model_name: Base name of the saved files
            
        Returns:
            Tuple containing:
            - Loaded XGBoost model
            - List of feature names
        """
        # Construct file paths
        model_file = os.path.join(save_path, f"{model_name}.json")
        features_file = os.path.join(save_path, f"{model_name}_features.json")
        
        # Check if files exist
        if not os.path.exists(model_file) or not os.path.exists(features_file):
            raise FileNotFoundError(f"Model files not found in {save_path}")
        
        # Load the model
        model = xgb.XGBRegressor()
        model.load_model(model_file)
        
        # Load feature names
        with open(features_file, 'r') as f:
            features_data = json.load(f)
        
        feature_names = features_data['feature_names']
        
        print(f"Model loaded from: {model_file}")
        print(f"Using XGBoost version: {features_data['xgboost_version']}")
        print(f"Number of features: {len(feature_names)}")
        
        return model, feature_names



def make_prediction(
    model: xgb.XGBRegressor,
    feature_names: List[str],
    input_data: pd.DataFrame,
    required_features: Optional[List[str]] = None
) -> pd.Series:
    """
    Make predictions using loaded model while ensuring feature alignment.
    
    Args:
        model: Loaded XGBoost model
        feature_names: List of feature names the model was trained with
        input_data: DataFrame containing input features
        required_features: Optional list of features that must be present
        
    Returns:
        Series containing predictions
    """
    # Verify required features if specified
    if required_features:
        missing_features = set(required_features) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
    
    # Ensure input data has all features in the correct order
    input_aligned = pd.DataFrame(index=input_data.index)
    for feature in feature_names:
        if feature not in input_data.columns:
            raise ValueError(f"Missing feature in input data: {feature}")
        input_aligned[feature] = input_data[feature]
    
    print("input aligned:", input_aligned)
    # Make predictions
    predictions = model.predict(input_aligned)

    print("pred inside fn:", predictions)
    
    return pd.Series(predictions, index=input_data.index)