import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import shap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class AdvancedOptimizer:
    def __init__(self):
        self.best_params = {}
        self.optimized_models = {}
        self.shap_values = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for optimization"""
        print("Loading and preprocessing data...")
        
        # Load data
        train_data = pd.read_csv('backend/train.csv')
        test_data = pd.read_csv('backend/test.csv')
        
        # Handle categorical variables
        categorical_features = ['Stage_fear', 'Drained_after_socializing']
        
        for feature in categorical_features:
            le = LabelEncoder()
            train_data[feature] = train_data[feature].fillna('Unknown')
            le.fit(train_data[feature].unique())
            train_data[feature] = le.transform(train_data[feature])
            
            test_data[feature] = test_data[feature].fillna('Unknown')
            test_data[feature] = le.transform(test_data[feature])
        
        # Handle numerical features
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        knn_imputer = KNNImputer(n_neighbors=7, weights='distance')
        train_imputed = knn_imputer.fit_transform(train_data[numerical_features])
        test_imputed = knn_imputer.transform(test_data[numerical_features])
        
        train_data[numerical_features] = train_imputed
        test_data[numerical_features] = test_imputed
        
        # Advanced feature engineering
        train_data['weighted_social_score'] = (
            train_data['Social_event_attendance'] * 0.4 + 
            train_data['Going_outside'] * 0.3 + 
            train_data['Post_frequency'] * 0.3
        )
        test_data['weighted_social_score'] = (
            test_data['Social_event_attendance'] * 0.4 + 
            test_data['Going_outside'] * 0.3 + 
            test_data['Post_frequency'] * 0.3
        )
        
        train_data['enhanced_introversion_score'] = (
            train_data['Time_spent_Alone'] * 0.4 + 
            train_data['Stage_fear'] * 0.3 + 
            train_data['Drained_after_socializing'] * 0.3
        )
        test_data['enhanced_introversion_score'] = (
            test_data['Time_spent_Alone'] * 0.4 + 
            test_data['Stage_fear'] * 0.3 + 
            test_data['Drained_after_socializing'] * 0.3
        )
        
        train_data['social_efficiency'] = train_data['weighted_social_score'] / (train_data['Friends_circle_size'] + 1)
        test_data['social_efficiency'] = test_data['weighted_social_score'] / (test_data['Friends_circle_size'] + 1)
        
        train_data['alone_ratio'] = train_data['Time_spent_Alone'] / (train_data['Time_spent_Alone'] + train_data['weighted_social_score'] + 1)
        test_data['alone_ratio'] = test_data['Time_spent_Alone'] / (test_data['Time_spent_Alone'] + test_data['weighted_social_score'] + 1)
        
        train_data['fear_social_interaction'] = train_data['Stage_fear'] * train_data['Social_event_attendance']
        test_data['fear_social_interaction'] = test_data['Stage_fear'] * test_data['Social_event_attendance']
        
        train_data['drain_social_interaction'] = train_data['Drained_after_socializing'] * train_data['Social_event_attendance']
        test_data['drain_social_interaction'] = test_data['Drained_after_socializing'] * test_data['Social_event_attendance']
        
        train_data['social_confidence'] = (1 - train_data['Stage_fear']) * train_data['Social_event_attendance']
        test_data['social_confidence'] = (1 - test_data['Stage_fear']) * test_data['Social_event_attendance']
        
        train_data['energy_balance'] = train_data['weighted_social_score'] - train_data['enhanced_introversion_score']
        test_data['energy_balance'] = test_data['weighted_social_score'] - test_data['enhanced_introversion_score']
        
        train_data['alone_social_interaction'] = train_data['Time_spent_Alone'] * train_data['weighted_social_score']
        test_data['alone_social_interaction'] = test_data['Time_spent_Alone'] * test_data['weighted_social_score']
        
        train_data['friends_activity_ratio'] = train_data['Friends_circle_size'] / (train_data['weighted_social_score'] + 1)
        test_data['friends_activity_ratio'] = test_data['Friends_circle_size'] / (test_data['weighted_social_score'] + 1)
        
        train_data['fear_drain_interaction'] = train_data['Stage_fear'] * train_data['Drained_after_socializing']
        test_data['fear_drain_interaction'] = test_data['Stage_fear'] * test_data['Drained_after_socializing']
        
        # Prepare features
        feature_columns = [
            'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency', 'weighted_social_score', 'enhanced_introversion_score',
            'social_efficiency', 'alone_ratio', 'fear_social_interaction', 
            'drain_social_interaction', 'social_confidence', 'energy_balance',
            'alone_social_interaction', 'friends_activity_ratio', 'fear_drain_interaction'
        ]
        
        X = train_data[feature_columns]
        y = train_data['Personality']
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.target_encoder = target_encoder
        self.test_data = test_data
        
        print("Data preprocessing completed!")
        
    def optimize_random_forest(self, trial):
        """Optimize Random Forest with Optuna"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='accuracy')
        return scores.mean()
        
    def optimize_xgboost(self, trial):
        """Optimize XGBoost with Optuna"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
        
        model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
        scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='accuracy')
        return scores.mean()
        
    def optimize_lightgbm(self, trial):
        """Optimize LightGBM with Optuna"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
        
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='accuracy')
        return scores.mean()
        
    def optimize_catboost(self, trial):
        """Optimize CatBoost with Optuna"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255)
        }
        
        model = cb.CatBoostClassifier(**params, random_state=42, verbose=False)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='accuracy')
        return scores.mean()
        
    def run_optuna_optimization(self):
        """Run Optuna optimization for all models"""
        print("\n=== OPTUNA HYPERPARAMETER OPTIMIZATION ===")
        
        models_to_optimize = {
            'RandomForest': self.optimize_random_forest,
            'XGBoost': self.optimize_xgboost,
            'LightGBM': self.optimize_lightgbm,
            'CatBoost': self.optimize_catboost
        }
        
        for model_name, objective_func in models_to_optimize.items():
            print(f"Optimizing {model_name}...")
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_func, n_trials=50)
            
            print(f"{model_name} - Best CV Score: {study.best_value:.4f}")
            print(f"{model_name} - Best Params: {study.best_params}")
            
            # Store best parameters
            self.best_params[model_name] = study.best_params
            
            # Train model with best parameters
            if model_name == 'RandomForest':
                model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
            elif model_name == 'XGBoost':
                model = xgb.XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss')
            elif model_name == 'LightGBM':
                model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
            elif model_name == 'CatBoost':
                model = cb.CatBoostClassifier(**study.best_params, random_state=42, verbose=False)
            
            model.fit(self.X_train, self.y_train)
            self.optimized_models[model_name] = model
            
    def create_advanced_ensemble(self):
        """Create advanced ensemble with optimized models"""
        print("\n=== CREATING ADVANCED ENSEMBLE ===")
        
        # Create base models list
        base_models = []
        for name, model in self.optimized_models.items():
            base_models.append((name, model))
        
        # Add linear models
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
        svm.fit(self.X_train_scaled, self.y_train)
        base_models.append(('SVM', svm))
        
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='liblinear')
        lr.fit(self.X_train_scaled, self.y_train)
        base_models.append(('LogisticRegression', lr))
        
        # Create stacking classifier
        meta_classifier = LogisticRegression(random_state=42)
        
        self.ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_classifier,
            cv=3,
            stack_method='predict_proba'
        )
        
        # Train ensemble
        self.ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = self.ensemble.predict(self.X_val)
        ensemble_accuracy = accuracy_score(self.y_val, y_pred_ensemble)
        
        print(f"Advanced Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        return ensemble_accuracy
        
    def analyze_shap_values(self):
        """Analyze SHAP values for model interpretability"""
        print("\n=== SHAP ANALYSIS ===")
        
        # Analyze SHAP values for the best model
        best_model_name = max(self.optimized_models.keys(), key=lambda x: self.optimized_models[x].score(self.X_val, self.y_val))
        best_model = self.optimized_models[best_model_name]
        
        print(f"Analyzing SHAP values for {best_model_name}...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(self.X_val)
        
        # Plot SHAP summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_val, feature_names=self.feature_columns, show=False)
        plt.title(f'SHAP Summary Plot - {best_model_name}')
        plt.tight_layout()
        plt.savefig('backend/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot SHAP bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_val, feature_names=self.feature_columns, plot_type='bar', show=False)
        plt.title(f'SHAP Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('backend/shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.shap_values[best_model_name] = shap_values
        
        print("SHAP analysis completed and plots saved!")
        
    def generate_optimized_predictions(self):
        """Generate predictions with optimized ensemble"""
        print("\n=== GENERATING OPTIMIZED PREDICTIONS ===")
        
        # Prepare test features
        X_test = self.test_data[self.feature_columns]
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions with ensemble
        y_pred_proba = self.ensemble.predict_proba(X_test)
        y_pred = self.target_encoder.inverse_transform(y_pred_proba.argmax(axis=1))
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Personality': y_pred
        })
        
        # Save predictions
        submission.to_csv('backend/advanced_optimized_submission.csv', index=False)
        print("Advanced optimized predictions saved to backend/advanced_optimized_submission.csv")
        
        # Display prediction distribution
        print("\nPrediction distribution:")
        print(submission['Personality'].value_counts())
        
        return submission
        
    def run_complete_optimization(self):
        """Run the complete advanced optimization pipeline"""
        print("=== ADVANCED OPTIMIZATION PIPELINE ===\n")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Run Optuna optimization
        self.run_optuna_optimization()
        
        # Create advanced ensemble
        ensemble_accuracy = self.create_advanced_ensemble()
        
        # Analyze SHAP values
        self.analyze_shap_values()
        
        # Generate predictions
        submission = self.generate_optimized_predictions()
        
        print("\n=== ADVANCED OPTIMIZATION COMPLETED ===")
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        return submission

if __name__ == "__main__":
    # Create and run the advanced optimizer
    optimizer = AdvancedOptimizer()
    submission = optimizer.run_complete_optimization() 