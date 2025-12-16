import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.cluster import DBSCAN
import xgboost as xgb
import joblib
import warnings
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveMaintenanceModel:
    def __init__(self, model_type='ensemble'):
        """
        Initialize predictive maintenance model
        
        Args:
            model_type: 'rf' for Random Forest, 'gb' for Gradient Boosting, 
                       'xgb' for XGBoost, 'ensemble' for ensemble of all
        """
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.thresholds = {
            'engine_failure': 0.7,
            'battery_failure': 0.65,
            'brake_failure': 0.75,
            'tire_failure': 0.6,
            'transmission_failure': 0.68
        }
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def generate_synthetic_data(self, n_samples=50000, include_failures=True):
        """
        Generate comprehensive synthetic telemetry data for training
        
        Args:
            n_samples: Number of samples to generate
            include_failures: Whether to include failure cases
        """
        logger.info(f"Generating {n_samples} synthetic data samples...")
        np.random.seed(42)
        
        # Vehicle characteristics
        makes = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Tesla', 'Hyundai', 'Kia']
        models = ['Sedan', 'SUV', 'Truck', 'Coupe', 'Hatchback', 'Convertible']
        fuel_types = ['Gasoline', 'Diesel', 'Hybrid', 'Electric']
        
        data = {
            'vehicle_id': [f'VIN{str(i).zfill(8)}' for i in range(n_samples)],
            'make': np.random.choice(makes, n_samples),
            'model': np.random.choice(models, n_samples),
            'year': np.random.randint(2018, 2024, n_samples),
            'fuel_type': np.random.choice(fuel_types, n_samples),
            'mileage': np.random.exponential(50000, n_samples).clip(max=200000),
            'days_since_service': np.random.exponential(90, n_samples).clip(max=365),
            'service_history_score': np.random.uniform(0.5, 1.0, n_samples),
            
            # Telemetry data
            'engine_temp': np.random.normal(90, 8, n_samples),
            'oil_pressure': np.random.normal(45, 4, n_samples),
            'oil_quality': np.random.uniform(0.3, 1.0, n_samples),  # 1 = fresh, 0 = degraded
            'battery_voltage': np.random.normal(12.6, 0.4, n_samples),
            'battery_health': np.random.uniform(0.4, 1.0, n_samples),  # State of Health
            'tire_pressure_front': np.random.normal(32, 1.5, n_samples),
            'tire_pressure_rear': np.random.normal(32, 1.5, n_samples),
            'tire_tread_depth': np.random.uniform(2, 10, n_samples),  # mm
            'brake_pad_thickness': np.random.uniform(3, 12, n_samples),  # mm
            'brake_fluid_level': np.random.uniform(0.7, 1.0, n_samples),
            'coolant_level': np.random.uniform(0.7, 1.0, n_samples),
            'transmission_temp': np.random.normal(85, 7, n_samples),
            'engine_rpm': np.random.normal(2000, 400, n_samples),
            'speed': np.random.exponential(40, n_samples).clip(max=120),
            'fuel_efficiency': np.random.normal(25, 5, n_samples),  # MPG or equivalent
            'vibration_level': np.random.exponential(0.5, n_samples).clip(max=3.0),
            'ambient_temp': np.random.normal(20, 10, n_samples),
            'driving_mode': np.random.choice(['city', 'highway', 'mixed'], n_samples),
            'load_capacity': np.random.uniform(0.1, 1.0, n_samples),  % of max capacity
        }
        
        df = pd.DataFrame(data)
        
        if include_failures:
            # Generate failure indicators based on realistic scenarios
            df = self._generate_failure_labels(df)
            
            # Add some anomaly patterns
            df = self._add_anomalies(df)
        
        logger.info(f"Synthetic data generated with {len(df.columns)} features")
        return df
    
    def _generate_failure_labels(self, df):
        """Generate realistic failure labels based on feature patterns"""
        
        # Engine failure: high temp + low oil pressure + old oil
        engine_failure_prob = (
            (df['engine_temp'] > 105).astype(int) * 0.3 +
            (df['oil_pressure'] < 35).astype(int) * 0.25 +
            (df['oil_quality'] < 0.4).astype(int) * 0.25 +
            (df['days_since_service'] > 180).astype(int) * 0.2
        )
        df['engine_failure'] = (np.random.random(len(df)) < engine_failure_prob).astype(int)
        
        # Battery failure: low voltage + poor health + extreme temps
        battery_failure_prob = (
            (df['battery_voltage'] < 12.0).astype(int) * 0.4 +
            (df['battery_health'] < 0.6).astype(int) * 0.3 +
            ((df['ambient_temp'] < 0) | (df['ambient_temp'] > 35)).astype(int) * 0.15 +
            (df['mileage'] > 100000).astype(int) * 0.15
        )
        df['battery_failure'] = (np.random.random(len(df)) < battery_failure_prob).astype(int)
        
        # Brake failure: worn pads + low fluid
        brake_failure_prob = (
            (df['brake_pad_thickness'] < 4).astype(int) * 0.5 +
            (df['brake_fluid_level'] < 0.8).astype(int) * 0.3 +
            (df['mileage'] > 80000).astype(int) * 0.2
        )
        df['brake_failure'] = (np.random.random(len(df)) < brake_failure_prob).astype(int)
        
        # Tire failure: uneven pressure + low tread
        tire_failure_prob = (
            (abs(df['tire_pressure_front'] - df['tire_pressure_rear']) > 3).astype(int) * 0.3 +
            (df['tire_tread_depth'] < 3).astype(int) * 0.4 +
            (df['vibration_level'] > 1.5).astype(int) * 0.3
        )
        df['tire_failure'] = (np.random.random(len(df)) < tire_failure_prob).astype(int)
        
        # Transmission failure: high temp + vibration
        transmission_failure_prob = (
            (df['transmission_temp'] > 100).astize(int) * 0.4 +
            (df['vibration_level'] > 2.0).astype(int) * 0.3 +
            (df['load_capacity'] > 0.8).astize(int) * 0.3
        )
        df['transmission_failure'] = (np.random.random(len(df)) < transmission_failure_prob).astype(int)
        
        # Overall vehicle health score (1-100)
        df['health_score'] = 100 - (
            df['engine_failure'] * 25 +
            df['battery_failure'] * 20 +
            df['brake_failure'] * 15 +
            df['tire_failure'] * 10 +
            df['transmission_failure'] * 10
        ).clip(upper=80)
        
        # Anomaly score (for unsupervised detection)
        df['anomaly_score'] = (
            (df['engine_temp'] > 110).astize(int) * 10 +
            (df['oil_pressure'] < 30).astize(int) * 15 +
            (df['battery_voltage'] < 11.8).astize(int) * 12 +
            (df['vibration_level'] > 2.5).astize(int) * 8
        )
        
        return df
    
    def _add_anomalies(self, df, anomaly_rate=0.05):
        """Add realistic anomalies to the data"""
        n_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Randomly select features to make anomalous
            anomaly_type = np.random.choice(['extreme_value', 'pattern_change', 'sensor_fault'])
            
            if anomaly_type == 'extreme_value':
                # Extreme values
                df.loc[idx, 'engine_temp'] = np.random.uniform(120, 140)
                df.loc[idx, 'vibration_level'] = np.random.uniform(2.5, 4.0)
                
            elif anomaly_type == 'pattern_change':
                # Unusual patterns
                df.loc[idx, 'engine_temp'] = 110
                df.loc[idx, 'engine_rpm'] = 500  # Very low RPM with high temp
                
            elif anomaly_type == 'sensor_fault':
                # Sensor faults (frozen values)
                df.loc[idx, 'oil_pressure'] = 45  # Stuck at normal value
                df.loc[idx, 'battery_voltage'] = 12.6  # Stuck at normal value
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess data for model training"""
        logger.info("Preprocessing data...")
        
        # Separate features and targets
        failure_columns = ['engine_failure', 'battery_failure', 'brake_failure', 
                          'tire_failure', 'transmission_failure']
        
        # Features for prediction
        feature_columns = [
            'mileage', 'days_since_service', 'service_history_score',
            'engine_temp', 'oil_pressure', 'oil_quality',
            'battery_voltage', 'battery_health',
            'tire_pressure_front', 'tire_pressure_rear', 'tire_tread_depth',
            'brake_pad_thickness', 'brake_fluid_level',
            'coolant_level', 'transmission_temp',
            'engine_rpm', 'speed', 'fuel_efficiency',
            'vibration_level', 'ambient_temp', 'load_capacity'
        ]
        
        # Encode categorical variables
        categorical_cols = ['make', 'model', 'fuel_type', 'driving_mode']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        X = df[feature_columns].copy()
        y = df[failure_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Preprocessed {len(X)} samples with {len(feature_columns)} features")
        return X_scaled, y, feature_columns
    
    def train_models(self, df=None):
        """Train models for different failure types"""
        if df is None:
            df = self.generate_synthetic_data(10000)
        
        X, y, feature_columns = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.sum(axis=1)
        )
        
        # Train anomaly detector
        logger.info("Training anomaly detector...")
        self.anomaly_detector.fit(X_train)
        
        # Train models for each failure type
        failure_types = y.columns.tolist()
        
        for failure in failure_types:
            logger.info(f"\nTraining model for {failure}...")
            
            y_train_failure = y_train[failure]
            y_test_failure = y_test[failure]
            
            # Handle class imbalance using class weights
            class_weights = {
                0: 1.0,
                1: len(y_train_failure[y_train_failure == 0]) / len(y_train_failure[y_train_failure == 1])
            }
            
            if self.model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight=class_weights,
                    random_state=42,
                    n_jobs=-1
                )
            elif self.model_type == 'gb':
                model = GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    random_state=42
                )
            elif self.model_type == 'xgb':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=class_weights[1],
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            else:  # ensemble
                # Train multiple models and use voting
                from sklearn.ensemble import VotingClassifier
                
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
                xgb_clf = xgb.XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False)
                
                model = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb), ('xgb', xgb_clf)],
                    voting='soft',
                    weights=[1, 1, 1]
                )
            
            # Train model
            model.fit(X_train, y_train_failure)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test_failure, y_pred)
            roc_auc = roc_auc_score(y_test_failure, y_pred_proba)
            
            logger.info(f"  Accuracy: {accuracy:.2%}")
            logger.info(f"  ROC AUC: {roc_auc:.2%}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                self.feature_importance[failure] = dict(zip(feature_columns, importances))
            
            # Store model
            self.models[failure] = model
            
            # Print detailed report for the first model
            if failure == failure_types[0]:
                logger.info("\n  Classification Report:")
                logger.info(classification_report(y_test_failure, y_pred))
                
                # Confusion matrix
                cm = confusion_matrix(y_test_failure, y_pred)
                logger.info(f"  Confusion Matrix:\n{cm}")
        
        # Save models
        self.save_models()
        
        # Train ensemble model for overall health prediction
        self._train_health_model(X_train, y_train, X_test, y_test)
        
        return self.models
    
    def _train_health_model(self, X_train, y_train, X_test, y_test):
        """Train a model for overall health score prediction"""
        logger.info("\nTraining overall health prediction model...")
        
        # Calculate health score from individual failures
        health_train = 100 - (y_train.sum(axis=1) * 20).clip(upper=80)
        health_test = 100 - (y_test.sum(axis=1) * 20).clip(upper=80)
        
        # Treat as regression problem
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        
        health_model = RandomForestRegressor(n_estimators=100, random_state=42)
        health_model.fit(X_train, health_train)
        
        # Predict
        health_pred = health_model.predict(X_test)
        
        mae = mean_absolute_error(health_test, health_pred)
        r2 = r2_score(health_test, health_pred)
        
        logger.info(f"  Health Score Prediction:")
        logger.info(f"  MAE: {mae:.2f} points")
        logger.info(f"  R² Score: {r2:.2%}")
        
        self.models['health_score'] = health_model
    
    def predict(self, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict failures for a single vehicle
        
        Args:
            vehicle_data: Dictionary containing vehicle telemetry data
            
        Returns:
            Dictionary with predictions for all failure types
        """
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([vehicle_data])
            
            # Preprocess (same as training)
            feature_columns = [
                'mileage', 'days_since_service', 'service_history_score',
                'engine_temp', 'oil_pressure', 'oil_quality',
                'battery_voltage', 'battery_health',
                'tire_pressure_front', 'tire_pressure_rear', 'tire_tread_depth',
                'brake_pad_thickness', 'brake_fluid_level',
                'coolant_level', 'transmission_temp',
                'engine_rpm', 'speed', 'fuel_efficiency',
                'vibration_level', 'ambient_temp', 'load_capacity'
            ]
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            
            # Select and order features
            X_input = input_df[feature_columns]
            
            # Handle missing values
            X_input = X_input.fillna(X_input.mean())
            
            # Scale features
            X_scaled = self.scaler.transform(X_input)
            
            # Detect anomalies
            anomaly_pred = self.anomaly_detector.predict(X_scaled)
            anomaly_score = self.anomaly_detector.decision_function(X_scaled)
            
            # Make predictions for each failure type
            predictions = {}
            
            for failure_type, model in self.models.items():
                if failure_type == 'health_score':
                    continue
                    
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0][1]
                else:
                    proba = model.predict(X_scaled)[0]
                
                # Determine severity based on probability and threshold
                threshold = self.thresholds.get(failure_type, 0.7)
                
                if proba >= threshold:
                    if proba >= threshold + 0.15:
                        severity = 'critical'
                    elif proba >= threshold + 0.05:
                        severity = 'high'
                    else:
                        severity = 'medium'
                else:
                    if proba >= threshold - 0.1:
                        severity = 'low'
                    else:
                        severity = 'none'
                
                predictions[failure_type] = {
                    'probability': float(proba),
                    'severity': severity,
                    'threshold': float(threshold),
                    'exceeds_threshold': proba >= threshold,
                    'recommended_action': self._get_recommendation(failure_type, severity, proba),
                    'confidence': float(self._calculate_confidence(proba, failure_type))
                }
            
            # Predict health score
            if 'health_score' in self.models:
                health_score = self.models['health_score'].predict(X_scaled)[0]
                health_score = max(0, min(100, health_score))
            else:
                # Calculate health score from individual predictions
                health_score = 100
                for failure, pred in predictions.items():
                    if pred['severity'] == 'critical':
                        health_score -= 25
                    elif pred['severity'] == 'high':
                        health_score -= 15
                    elif pred['severity'] == 'medium':
                        health_score -= 8
                    elif pred['severity'] == 'low':
                        health_score -= 3
                health_score = max(0, health_score)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(predictions)
            
            # Feature contributions (explainability)
            top_features = self._get_top_contributing_features(X_scaled[0], feature_columns)
            
            result = {
                'predictions': predictions,
                'health_score': float(health_score),
                'risk_score': float(risk_score),
                'anomaly_detected': anomaly_pred[0] == -1,
                'anomaly_score': float(anomaly_score[0]),
                'top_contributing_factors': top_features,
                'timestamp': datetime.now().isoformat(),
                'model_versions': {k: '1.0' for k in self.models.keys()}
            }
            
            logger.info(f"Predictions generated. Health: {health_score:.1f}, Risk: {risk_score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Return fallback predictions
            return self._get_fallback_predictions()
    
    def _calculate_risk_score(self, predictions: Dict) -> float:
        """Calculate overall risk score (0-100)"""
        risk = 0
        weights = {'critical': 5, 'high': 3, 'medium': 2, 'low': 1, 'none': 0}
        
        for failure, pred in predictions.items():
            severity = pred.get('severity', 'none')
            probability = pred.get('probability', 0)
            risk += weights.get(severity, 0) * probability * 20
        
        return min(100, risk)
    
    def _get_recommendation(self, failure_type: str, severity: str, probability: float) -> str:
        """Get recommendation based on failure type and severity"""
        recommendations = {
            'engine_failure': {
                'critical': 'Immediate engine shutdown and tow to service center',
                'high': 'Schedule engine inspection within 24 hours',
                'medium': 'Monitor engine temperature and schedule service',
                'low': 'Check coolant and oil levels'
            },
            'battery_failure': {
                'critical': 'Replace battery immediately',
                'high': 'Schedule battery replacement within 3 days',
                'medium': 'Test battery and charging system',
                'low': 'Monitor battery voltage'
            },
            'brake_failure': {
                'critical': 'Stop driving immediately, tow required',
                'high': 'Schedule brake service within 2 days',
                'medium': 'Inspect brakes at next service',
                'low': 'Monitor brake performance'
            },
            'tire_failure': {
                'critical': 'Replace tires immediately',
                'high': 'Schedule tire replacement within 7 days',
                'medium': 'Check tire pressure and alignment',
                'low': 'Monitor tire wear'
            },
            'transmission_failure': {
                'critical': 'Stop driving, transmission service required',
                'high': 'Schedule transmission check within 3 days',
                'medium': 'Check transmission fluid',
                'low': 'Monitor transmission performance'
            }
        }
        
        failure_recs = recommendations.get(failure_type, {})
        return failure_recs.get(severity, 'No action required at this time')
    
    def _calculate_confidence(self, probability: float, failure_type: str) -> float:
        """Calculate confidence score for prediction"""
        # Higher confidence for extreme probabilities
        if probability > 0.9 or probability < 0.1:
            return 0.95
        elif probability > 0.7 or probability < 0.3:
            return 0.85
        else:
            # Medium probabilities are less certain
            return 0.75
    
    def _get_top_contributing_features(self, features: np.ndarray, feature_names: List[str]) -> List[Dict]:
        """Get top contributing features for the prediction"""
        if not self.feature_importance:
            return []
        
        # Calculate contribution scores
        contributions = []
        for failure_type, importance_dict in self.feature_importance.items():
            for feature_name, importance in importance_dict.items():
                idx = feature_names.index(feature_name)
                value = features[idx]
                # Normalized contribution
                contribution = importance * abs(value - self.scaler.mean_[idx]) / self.scaler.scale_[idx]
                contributions.append({
                    'feature': feature_name,
                    'importance': float(importance),
                    'value': float(value),
                    'contribution': float(contribution),
                    'failure_type': failure_type
                })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Return top 5
        return contributions[:5]
    
    def _get_fallback_predictions(self) -> Dict:
        """Return fallback predictions when model fails"""
        return {
            'predictions': {
                'engine_failure': {'probability': 0.1, 'severity': 'low', 'recommended_action': 'System error'},
                'battery_failure': {'probability': 0.1, 'severity': 'low', 'recommended_action': 'System error'},
                'brake_failure': {'probability': 0.1, 'severity': 'low', 'recommended_action': 'System error'},
                'tire_failure': {'probability': 0.1, 'severity': 'low', 'recommended_action': 'System error'},
                'transmission_failure': {'probability': 0.1, 'severity': 'low', 'recommended_action': 'System error'}
            },
            'health_score': 80.0,
            'risk_score': 20.0,
            'anomaly_detected': False,
            'anomaly_score': 0.0,
            'top_contributing_factors': [],
            'timestamp': datetime.now().isoformat(),
            'model_versions': {'fallback': '1.0'}
        }
    
    def save_models(self, path='models/'):
        """Save trained models to disk"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f'{path}/{name}_model.pkl')
        
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.anomaly_detector, f'{path}/anomaly_detector.pkl')
        
        # Save thresholds and feature importance
        metadata = {
            'thresholds': self.thresholds,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{path}/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path='models/'):
        """Load trained models from disk"""
        import glob
        import os
        
        # Load models
        model_files = glob.glob(f'{path}/*_model.pkl')
        for file in model_files:
            name = os.path.basename(file).replace('_model.pkl', '')
            self.models[name] = joblib.load(file)
        
        # Load other components
        if os.path.exists(f'{path}/scaler.pkl'):
            self.scaler = joblib.load(f'{path}/scaler.pkl')
        
        if os.path.exists(f'{path}/anomaly_detector.pkl'):
            self.anomaly_detector = joblib.load(f'{path}/anomaly_detector.pkl')
        
        # Load metadata
        if os.path.exists(f'{path}/model_metadata.json'):
            with open(f'{path}/model_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.thresholds = metadata.get('thresholds', self.thresholds)
                self.feature_importance = metadata.get('feature_importance', {})
        
        logger.info(f"Loaded {len(self.models)} models from {path}")
        return self
    
    def evaluate_on_real_data(self, real_data: pd.DataFrame):
        """Evaluate model performance on real-world data"""
        logger.info("Evaluating on real data...")
        
        # Preprocess real data
        X_real, y_real, _ = self.preprocess_data(real_data)
        
        results = {}
        
        for failure_type, model in self.models.items():
            if failure_type == 'health_score':
                continue
                
            if failure_type in y_real.columns:
                y_pred = model.predict(X_real)
                y_pred_proba = model.predict_proba(X_real)[:, 1]
                
                accuracy = accuracy_score(y_real[failure_type], y_pred)
                roc_auc = roc_auc_score(y_real[failure_type], y_pred_proba)
                
                results[failure_type] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'n_samples': len(X_real),
                    'positive_rate': y_real[failure_type].mean()
                }
        
        return results

# Example usage and testing
def main():
    """Main function to demonstrate the predictive maintenance model"""
    
    # Initialize model
    logger.info("Initializing Predictive Maintenance Model...")
    pm_model = PredictiveMaintenanceModel(model_type='ensemble')
    
    # Option 1: Train new models
    logger.info("\n" + "="*60)
    logger.info("TRAINING NEW MODELS")
    logger.info("="*60)
    pm_model.train_models()
    
    # Option 2: Or load pre-trained models
    # pm_model.load_models('models/')
    
    # Test prediction with sample data
    logger.info("\n" + "="*60)
    logger.info("TESTING PREDICTIONS")
    logger.info("="*60)
    
    sample_vehicle = {
        'mileage': 85000,
        'days_since_service': 120,
        'service_history_score': 0.8,
        'engine_temp': 112,  # High temperature
        'oil_pressure': 32,  # Low pressure
        'oil_quality': 0.4,
        'battery_voltage': 11.9,  # Low voltage
        'battery_health': 0.5,
        'tire_pressure_front': 30,
        'tire_pressure_rear': 29,
        'tire_tread_depth': 4.5,
        'brake_pad_thickness': 5.0,
        'brake_fluid_level': 0.85,
        'coolant_level': 0.9,
        'transmission_temp': 95,
        'engine_rpm': 2200,
        'speed': 75,
        'fuel_efficiency': 22,
        'vibration_level': 1.8,
        'ambient_temp': 25,
        'load_capacity': 0.6
    }
    
    predictions = pm_model.predict(sample_vehicle)
    
    # Print results
    logger.info("\nPrediction Results:")
    logger.info(f"Health Score: {predictions['health_score']:.1f}/100")
    logger.info(f"Risk Score: {predictions['risk_score']:.1f}/100")
    logger.info(f"Anomaly Detected: {predictions['anomaly_detected']}")
    
    logger.info("\nComponent Predictions:")
    for component, pred in predictions['predictions'].items():
        logger.info(f"  {component.replace('_', ' ').title()}:")
        logger.info(f"    Probability: {pred['probability']:.1%}")
        logger.info(f"    Severity: {pred['severity']}")
        logger.info(f"    Action: {pred['recommended_action']}")
    
    if predictions['top_contributing_factors']:
        logger.info("\nTop Contributing Factors:")
        for factor in predictions['top_contributing_factors'][:3]:
            logger.info(f"  {factor['feature']}: {factor['value']:.2f} (impact: {factor['contribution']:.3f})")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Predictive Maintenance Model Test Complete")
    logger.info("="*60)
    
    return pm_model, predictions

if __name__ == "__main__":
    # Run the main function
    model, results = main()
    
    # You can save the results to a file
    with open('prediction_results.json', 'w') as f:
        import json
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            else:
                return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    print("\nResults saved to 'prediction_results.json'")