import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class PersonalityPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load and explore the dataset"""
        print("Loading data...")
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        # Display basic info
        print("\nTraining data info:")
        print(self.train_data.info())
        print("\nMissing values in training data:")
        print(self.train_data.isnull().sum())
        
        # Target distribution
        print("\nTarget distribution:")
        print(self.train_data['Personality'].value_counts())
        
    def explore_data(self):
        """Exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Target distribution
        plt.subplot(2, 3, 1)
        self.train_data['Personality'].value_counts().plot(kind='bar')
        plt.title('Target Distribution')
        plt.ylabel('Count')
        
        # Numerical features distribution
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        for i, feature in enumerate(numerical_features):
            plt.subplot(2, 3, i+2)
            self.train_data[feature].hist(bins=20, alpha=0.7)
            plt.title(f'{feature} Distribution')
            
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation analysis
        print("\nCorrelation with target:")
        for feature in numerical_features:
            correlation = self.train_data[feature].corr(self.train_data['Personality'].map({'Introvert': 0, 'Extrovert': 1}))
            print(f"{feature}: {correlation:.3f}")
            
    def preprocess_data(self):
        """Advanced data preprocessing"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Create copies for preprocessing
        train_processed = self.train_data.copy()
        test_processed = self.test_data.copy()
        
        # Handle categorical variables
        categorical_features = ['Stage_fear', 'Drained_after_socializing']
        
        for feature in categorical_features:
            # Create label encoders
            le = LabelEncoder()
            
            # Fit on training data only
            train_processed[feature] = train_processed[feature].fillna('Unknown')
            le.fit(train_processed[feature].unique())
            
            # Transform both datasets
            train_processed[feature] = le.transform(train_processed[feature])
            test_processed[feature] = test_processed[feature].fillna('Unknown')
            test_processed[feature] = le.transform(test_processed[feature])
            
            self.label_encoders[feature] = le
            
        # Handle missing values in numerical features
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        # Use KNN imputation for better results
        knn_imputer = KNNImputer(n_neighbors=5)
        
        # Fit on training data
        train_numerical = train_processed[numerical_features]
        train_imputed = knn_imputer.fit_transform(train_numerical)
        train_processed[numerical_features] = train_imputed
        
        # Transform test data
        test_numerical = test_processed[numerical_features]
        test_imputed = knn_imputer.transform(test_numerical)
        test_processed[numerical_features] = test_imputed
        
        # Feature engineering
        print("Creating engineered features...")
        
        # Social activity score
        train_processed['social_activity_score'] = (
            train_processed['Social_event_attendance'] + 
            train_processed['Going_outside'] + 
            train_processed['Post_frequency']
        )
        test_processed['social_activity_score'] = (
            test_processed['Social_event_attendance'] + 
            test_processed['Going_outside'] + 
            test_processed['Post_frequency']
        )
        
        # Introversion indicators
        train_processed['introversion_score'] = (
            train_processed['Time_spent_Alone'] + 
            train_processed['Stage_fear'] * 5 + 
            train_processed['Drained_after_socializing'] * 3
        )
        test_processed['introversion_score'] = (
            test_processed['Time_spent_Alone'] + 
            test_processed['Stage_fear'] * 5 + 
            test_processed['Drained_after_socializing'] * 3
        )
        
        # Social network size vs activity ratio
        train_processed['social_efficiency'] = train_processed['social_activity_score'] / (train_processed['Friends_circle_size'] + 1)
        test_processed['social_efficiency'] = test_processed['social_activity_score'] / (test_processed['Friends_circle_size'] + 1)
        
        # Alone time ratio
        train_processed['alone_ratio'] = train_processed['Time_spent_Alone'] / (train_processed['Time_spent_Alone'] + train_processed['social_activity_score'] + 1)
        test_processed['alone_ratio'] = test_processed['Time_spent_Alone'] / (test_processed['Time_spent_Alone'] + test_processed['social_activity_score'] + 1)
        
        # Feature interactions
        train_processed['fear_social_interaction'] = train_processed['Stage_fear'] * train_processed['Social_event_attendance']
        test_processed['fear_social_interaction'] = test_processed['Stage_fear'] * test_processed['Social_event_attendance']
        
        train_processed['drain_social_interaction'] = train_processed['Drained_after_socializing'] * train_processed['Social_event_attendance']
        test_processed['drain_social_interaction'] = test_processed['Drained_after_socializing'] * test_processed['Social_event_attendance']
        
        # Store processed data
        self.train_processed = train_processed
        self.test_processed = test_processed
        
        print("Data preprocessing completed!")
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n=== FEATURE PREPARATION ===")
        
        # Select features for modeling
        feature_columns = [
            'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency', 'social_activity_score', 'introversion_score',
            'social_efficiency', 'alone_ratio', 'fear_social_interaction', 
            'drain_social_interaction'
        ]
        
        # Prepare training data
        X = self.train_processed[feature_columns]
        y = self.train_processed['Personality']
        
        # Encode target variable
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
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        
    def train_models(self):
        """Train multiple models with hyperparameter tuning"""
        print("\n=== MODEL TRAINING ===")
        
        # Define models with optimized parameters
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                min_samples_split=5, random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbose=-1
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=200, learning_rate=0.1, depth=8,
                l2_leaf_reg=3, random_state=42, verbose=False
            ),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True),
            'LogisticRegression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, solver='liblinear'
            ),
            'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            'NaiveBayes': GaussianNB()
        }
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_val_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_val)
            
            accuracy = accuracy_score(self.y_val, y_pred)
            model_scores[name] = accuracy
            self.models[name] = model
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
        # Display model comparison
        print("\nModel Performance Comparison:")
        for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {score:.4f}")
            
        return model_scores
        
    def create_ensemble(self, top_models):
        """Create ensemble from top performing models"""
        print(f"\n=== CREATING ENSEMBLE FROM TOP {len(top_models)} MODELS ===")
        
        # Get top models
        ensemble_models = []
        for model_name in top_models:
            if model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes']:
                ensemble_models.append((model_name, self.models[model_name]))
            else:
                ensemble_models.append((model_name, self.models[model_name]))
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use probability predictions
        )
        
        # Train ensemble
        if any(model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes'] for model_name in top_models):
            ensemble.fit(self.X_train_scaled, self.y_train)
        else:
            ensemble.fit(self.X_train, self.y_train)
            
        # Evaluate ensemble
        if any(model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes'] for model_name in top_models):
            y_pred_ensemble = ensemble.predict(self.X_val_scaled)
        else:
            y_pred_ensemble = ensemble.predict(self.X_val)
            
        ensemble_accuracy = accuracy_score(self.y_val, y_pred_ensemble)
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        self.ensemble = ensemble
        return ensemble_accuracy
        
    def cross_validate_ensemble(self):
        """Perform cross-validation on the ensemble"""
        print("\n=== CROSS-VALIDATION ===")
        
        # Prepare full dataset
        X_full = self.train_processed[self.feature_columns]
        y_full = self.target_encoder.transform(self.train_processed['Personality'])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        if hasattr(self, 'ensemble'):
            if any(model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes'] 
                   for model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes']):
                cv_scores = cross_val_score(self.ensemble, self.scaler.transform(X_full), y_full, cv=cv, scoring='accuracy')
            else:
                cv_scores = cross_val_score(self.ensemble, X_full, y_full, cv=cv, scoring='accuracy')
        else:
            # Use best single model
            best_model_name = max(self.model_scores, key=self.model_scores.get)
            best_model = self.models[best_model_name]
            
            if best_model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes']:
                cv_scores = cross_val_score(best_model, self.scaler.transform(X_full), y_full, cv=cv, scoring='accuracy')
            else:
                cv_scores = cross_val_score(best_model, X_full, y_full, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    def generate_predictions(self):
        """Generate predictions for test set"""
        print("\n=== GENERATING PREDICTIONS ===")
        
        # Prepare test features
        X_test = self.test_processed[self.feature_columns]
        
        # Make predictions
        if hasattr(self, 'ensemble'):
            if any(model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes'] 
                   for model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes']):
                y_pred_proba = self.ensemble.predict_proba(self.scaler.transform(X_test))
            else:
                y_pred_proba = self.ensemble.predict_proba(X_test)
        else:
            # Use best single model
            best_model_name = max(self.model_scores, key=self.model_scores.get)
            best_model = self.models[best_model_name]
            
            if best_model_name in ['SVM', 'LogisticRegression', 'KNN', 'NaiveBayes']:
                y_pred_proba = best_model.predict_proba(self.scaler.transform(X_test))
            else:
                y_pred_proba = best_model.predict_proba(X_test)
        
        # Convert probabilities to predictions
        y_pred = self.target_encoder.inverse_transform(y_pred_proba.argmax(axis=1))
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Personality': y_pred
        })
        
        # Save predictions
        submission.to_csv('submission.csv', index=False)
        print("Predictions saved to submission.csv")
        
        # Display prediction distribution
        print("\nPrediction distribution:")
        print(submission['Personality'].value_counts())
        
        return submission
        
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Get feature importance from tree-based models
        importance_data = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = model.feature_importances_
        
        if importance_data:
            # Create feature importance plot
            plt.figure(figsize=(12, 8))
            
            # Average importance across models
            avg_importance = np.mean(list(importance_data.values()), axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': avg_importance
            }).sort_values('importance', ascending=True)
            
            plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
            plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Average Feature Importance Across Models')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Top 5 most important features:")
            for _, row in feature_importance_df.tail(5).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
                
    def run_complete_pipeline(self):
        """Run the complete modeling pipeline"""
        print("=== PERSONALITY PREDICTION PIPELINE ===\n")
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Prepare features
        self.prepare_features()
        
        # Train models
        self.model_scores = self.train_models()
        
        # Create ensemble from top 5 models
        top_models = sorted(self.model_scores.keys(), key=lambda x: self.model_scores[x], reverse=True)[:5]
        ensemble_accuracy = self.create_ensemble(top_models)
        
        # Cross-validation
        self.cross_validate_ensemble()
        
        # Feature importance analysis
        self.analyze_feature_importance()
        
        # Generate predictions
        submission = self.generate_predictions()
        
        print("\n=== PIPELINE COMPLETED ===")
        print(f"Best single model accuracy: {max(self.model_scores.values()):.4f}")
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        return submission

if __name__ == "__main__":
    # Create and run the predictor
    predictor = PersonalityPredictor()
    submission = predictor.run_complete_pipeline() 