import os
import pickle
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

logger = logging.getLogger(__name__)

class SkillGapMLModel:
    """Machine Learning model to predict skill gap severity based on semantic and graph features."""
    
    def __init__(self, model_path="data/ml_model.pkl"):
        self.model_path = model_path
        self.model = None
        
        # We try to load a pre-trained model. If it doesn't exist, we fallback to a mock model or train a dummy one.
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded pre-trained ML model for gap prediction.")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}. Falling back to dynamic mock model.")
                self._train_dummy_model()
        else:
            self._train_dummy_model()

    def _train_dummy_model(self):
        """Trains a dummy XGBoost Regressor model so the hybrid system has functional ML predictions."""
        logger.info("Training a dummy XGBoost model for severity prediction...")
        
        # Generate synthetic data for training
        # Features: [semantic_score, graph_score, missing_skills_count, total_required_count]
        np.random.seed(42)
        X_train = np.random.rand(500, 4)
        
        # We manually design the target `gap_severity_score` (0.0 to 1.0, where 1.0 means severe gap)
        # Higher semantic score -> lower gap
        # Higher missing count -> higher gap
        Y_train = (1.0 - X_train[:, 0]) * 0.4 + (1.0 - X_train[:, 1]) * 0.3 + (X_train[:, 2]) * 0.3
        Y_train = np.clip(Y_train + np.random.normal(0, 0.05, 500), 0, 1)

        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=3)
        self.model.fit(X_train, Y_train)
        
        # Make sure data directory exists (guard against empty dirname)
        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as e:
            logger.warning(f"Could not save dummy model: {e}")

    def predict_severity(self, semantic_score: float, graph_score: float, missing_count: int, total_required: int) -> float:
        """
        Predicts the continuous severity score of the skill gap (0 to 1).
        Returns a value where 1.0 = Severe Gap, 0.0 = No Gap.
        """
        # Normalize counts to 0-1 range for the model
        norm_missing = min(1.0, missing_count / max(1, total_required))
        norm_total = min(1.0, total_required / 20.0) # assume 20 is a high number of required skills
        
        features = np.array([[semantic_score, graph_score, norm_missing, norm_total]])
        prediction = self.model.predict(features)[0]
        
        return float(np.clip(prediction, 0.0, 1.0))

# Global instance
ml_model = SkillGapMLModel()
