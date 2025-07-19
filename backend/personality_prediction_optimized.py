import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class OptimizedPersonalityPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.best_params = {}
        
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
        """Enhanced exploratory data analysis"""
        print("\n=== ENHANCED EXPLORATORY DATA ANALYSIS ===")
        
        # Create visualizations
        plt.figure(figsize=(20, 15))
        
        # Target distribution
        plt.subplot(3, 4, 1)
        self.train_data['Personality'].value_counts().plot(kind='bar')
        plt.title('Target Distribution')
        plt.ylabel('Count')
        
        # Numerical features distribution
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        for i, feature in enumerate(numerical_features):
            plt.subplot(3, 4, i+2)
            self.train_data[feature].hist(bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{feature} Distribution')
            
        # Correlation heatmap
        plt.subplot(3, 4, 7)
        correlation_matrix = self.train_data[numerical_features + ['Personality']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        # Box plots for numerical features by personality
        for i, feature in enumerate(numerical_features):
            plt.subplot(3, 4, i+8)
            self.train_data.boxplot(column=feature, by='Personality', ax=plt.gca())
            plt.title(f'{feature} by Personality')
            plt.suptitle('')  # Remove default title
            
        plt.tight_layout()
        plt.savefig('enhanced_data_exploration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical analysis
        print("\nStatistical Analysis:")
        for feature in numerical_features:
            introvert_mean = self.train_data[self.train_data['Personality'] == 'Introvert'][feature].mean()
            extrovert_mean = self.train_data[self.train_data['Personality'] == 'Extrovert'][feature].mean()
            print(f"{feature}: Introvert={introvert_mean:.2f}, Extrovert={extrovert_mean:.2f}")
            
    def advanced_preprocess_data(self):
        """Advanced data preprocessing with sophisticated feature engineering"""
        print("\n=== ADVANCED DATA PREPROCESSING ===")
        
        # Create copies for preprocessing
        train_processed = self.train_data.copy()
        test_processed = self.test_data.copy()
        
        # Handle categorical variables with more sophisticated encoding
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
            
        # Handle missing values in numerical features with advanced imputation
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        # Use KNN imputation with optimized parameters
        knn_imputer = KNNImputer(n_neighbors=7, weights='distance')
        
        # Fit on training data
        train_numerical = train_processed[numerical_features]
        train_imputed = knn_imputer.fit_transform(train_numerical)
        train_processed[numerical_features] = train_imputed
        
        # Transform test data
        test_numerical = test_processed[numerical_features]
        test_imputed = knn_imputer.transform(test_numerical)
        test_processed[numerical_features] = test_imputed
        
        # Advanced feature engineering
        print("Creating advanced engineered features...")
        
        # Social activity score (weighted)
        train_processed['weighted_social_score'] = (
            train_processed['Social_event_attendance'] * 0.4 + 
            train_processed['Going_outside'] * 0.3 + 
            train_processed['Post_frequency'] * 0.3
        )
        test_processed['weighted_social_score'] = (
            test_processed['Social_event_attendance'] * 0.4 + 
            test_processed['Going_outside'] * 0.3 + 
            test_processed['Post_frequency'] * 0.3
        )
        
        # Enhanced introversion indicators
        train_processed['enhanced_introversion_score'] = (
            train_processed['Time_spent_Alone'] * 0.4 + 
            train_processed['Stage_fear'] * 0.3 + 
            train_processed['Drained_after_socializing'] * 0.3
        )
        test_processed['enhanced_introversion_score'] = (
            test_processed['Time_spent_Alone'] * 0.4 + 
            test_processed['Stage_fear'] * 0.3 + 
            test_processed['Drained_after_socializing'] * 0.3
        )
        
        # Social efficiency metrics
        train_processed['social_efficiency'] = train_processed['weighted_social_score'] / (train_processed['Friends_circle_size'] + 1)
        test_processed['social_efficiency'] = test_processed['weighted_social_score'] / (test_processed['Friends_circle_size'] + 1)
        
        # Alone time ratio
        train_processed['alone_ratio'] = train_processed['Time_spent_Alone'] / (train_processed['Time_spent_Alone'] + train_processed['weighted_social_score'] + 1)
        test_processed['alone_ratio'] = test_processed['Time_spent_Alone'] / (test_processed['Time_spent_Alone'] + test_processed['weighted_social_score'] + 1)
        
        # Advanced interaction features
        train_processed['fear_social_interaction'] = train_processed['Stage_fear'] * train_processed['Social_event_attendance']
        test_processed['fear_social_interaction'] = test_processed['Stage_fear'] * test_processed['Social_event_attendance']
        
        train_processed['drain_social_interaction'] = train_processed['Drained_after_socializing'] * train_processed['Social_event_attendance']
        test_processed['drain_social_interaction'] = test_processed['Drained_after_socializing'] * test_processed['Social_event_attendance']
        
        # New advanced features
        train_processed['social_confidence'] = (1 - train_processed['Stage_fear']) * train_processed['Social_event_attendance']
        test_processed['social_confidence'] = (1 - test_processed['Stage_fear']) * test_processed['Social_event_attendance']
        
        train_processed['energy_balance'] = train_processed['weighted_social_score'] - train_processed['enhanced_introversion_score']
        test_processed['energy_balance'] = test_processed['weighted_social_score'] - test_processed['enhanced_introversion_score']
        
        # Polynomial features for key interactions
        train_processed['alone_social_interaction'] = train_processed['Time_spent_Alone'] * train_processed['weighted_social_score']
        test_processed['alone_social_interaction'] = test_processed['Time_spent_Alone'] * test_processed['weighted_social_score']
        
        # Ratio features
        train_processed['friends_activity_ratio'] = train_processed['Friends_circle_size'] / (train_processed['weighted_social_score'] + 1)
        test_processed['friends_activity_ratio'] = test_processed['Friends_circle_size'] / (test_processed['weighted_social_score'] + 1)
        
        # Categorical interaction features
        train_processed['fear_drain_interaction'] = train_processed['Stage_fear'] * train_processed['Drained_after_socializing']
        test_processed['fear_drain_interaction'] = test_processed['Stage_fear'] * test_processed['Drained_after_socializing']
        
        # Store processed data
        self.train_processed = train_processed
        self.test_processed = test_processed
        
        print("Advanced data preprocessing completed!")
        
    def prepare_features_with_selection(self):
        """Prepare features with advanced feature selection"""
        print("\n=== ADVANCED FEATURE PREPARATION ===")
        
        # Select comprehensive feature set
        feature_columns = [
            'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency', 'weighted_social_score', 'enhanced_introversion_score',
            'social_efficiency', 'alone_ratio', 'fear_social_interaction', 
            'drain_social_interaction', 'social_confidence', 'energy_balance',
            'alone_social_interaction', 'friends_activity_ratio', 'fear_drain_interaction'
        ]
        
        # Prepare training data
        X = self.train_processed[feature_columns]
        y = self.train_processed['Personality']
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Advanced scaling with RobustScaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Feature selection using multiple methods
        print("Performing feature selection...")
        
        # Method 1: SelectKBest
        selector_kbest = SelectKBest(score_func=f_classif, k=12)
        X_train_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
        X_val_kbest = selector_kbest.transform(X_val_scaled)
        
        # Method 2: SelectFromModel with RandomForest
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector_rf = SelectFromModel(rf_selector, prefit=False)
        X_train_rf = selector_rf.fit_transform(X_train_scaled, y_train)
        X_val_rf = selector_rf.transform(X_val_scaled)
        
        # Choose the better feature selection method
        rf_selector.fit(X_train_scaled, y_train)
        kbest_score = rf_selector.score(X_val_kbest, y_val)
        rf_score = rf_selector.score(X_val_rf, y_val)
        
        if kbest_score > rf_score:
            self.X_train = X_train_kbest
            self.X_val = X_val_kbest
            self.feature_selector = selector_kbest
            print(f"Selected SelectKBest features (score: {kbest_score:.4f})")
        else:
            self.X_train = X_train_rf
            self.X_val = X_val_rf
            self.feature_selector = selector_rf
            print(f"Selected RandomForest features (score: {rf_score:.4f})")
        
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.target_encoder = target_encoder
        self.y_train = y_train
        self.y_val = y_val
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        
    def hyperparameter_tuning(self):
        """Advanced hyperparameter tuning for all models"""
        print("\n=== ADVANCED HYPERPARAMETER TUNING ===")
        
        # Define parameter grids for each model
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            'CatBoost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'depth': [6, 8, 10],
                'l2_leaf_reg': [1, 3, 5, 7],
                'border_count': [32, 64, 128]
            }
        }
        
        # Base models
        base_models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=False)
        }
        
        # Perform hyperparameter tuning
        for name, model in base_models.items():
            print(f"Tuning {name}...")
            
            # Use RandomizedSearchCV for faster tuning
            random_search = RandomizedSearchCV(
                model, param_grids[name], n_iter=20, cv=3, 
                scoring='accuracy', random_state=42, n_jobs=-1
            )
            
            random_search.fit(self.X_train, self.y_train)
            
            self.models[name] = random_search.best_estimator_
            self.best_params[name] = random_search.best_params_
            
            # Evaluate on validation set
            y_pred = random_search.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)
            
            print(f"{name} - Best params: {random_search.best_params_}")
            print(f"{name} - Best CV score: {random_search.best_score_:.4f}")
            print(f"{name} - Validation accuracy: {accuracy:.4f}")
            
        # For linear models, use scaled data
        linear_models = {
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        linear_param_grids = {
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2']
            }
        }
        
        for name, model in linear_models.items():
            print(f"Tuning {name}...")
            
            random_search = RandomizedSearchCV(
                model, linear_param_grids[name], n_iter=15, cv=3,
                scoring='accuracy', random_state=42, n_jobs=-1
            )
            
            random_search.fit(self.X_train_scaled, self.y_train)
            
            self.models[name] = random_search.best_estimator_
            self.best_params[name] = random_search.best_params_
            
            y_pred = random_search.predict(self.X_val_scaled)
            accuracy = accuracy_score(self.y_val, y_pred)
            
            print(f"{name} - Best params: {random_search.best_params_}")
            print(f"{name} - Best CV score: {random_search.best_score_:.4f}")
            print(f"{name} - Validation accuracy: {accuracy:.4f}")
            
    def create_advanced_ensemble(self):
        """Create advanced ensemble with stacking"""
        print("\n=== CREATING ADVANCED ENSEMBLE ===")
        
        # Get all trained models
        base_models = []
        for name, model in self.models.items():
            if name in ['SVM', 'LogisticRegression']:
                base_models.append((name, model))
            else:
                base_models.append((name, model))
        
        # Create stacking classifier
        meta_classifier = LogisticRegression(random_state=42)
        
        self.ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_classifier,
            cv=3,
            stack_method='predict_proba'
        )
        
        # Train ensemble
        if any(name in ['SVM', 'LogisticRegression'] for name in self.models.keys()):
            self.ensemble.fit(self.X_train_scaled, self.y_train)
            y_pred_ensemble = self.ensemble.predict(self.X_val_scaled)
        else:
            self.ensemble.fit(self.X_train, self.y_train)
            y_pred_ensemble = self.ensemble.predict(self.X_val)
            
        ensemble_accuracy = accuracy_score(self.y_val, y_pred_ensemble)
        print(f"Advanced Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        return ensemble_accuracy
        
    def advanced_cross_validation(self):
        """Perform advanced cross-validation with nested CV"""
        print("\n=== ADVANCED CROSS-VALIDATION ===")
        
        # Prepare full dataset
        X_full = self.train_processed[self.feature_columns]
        y_full = self.target_encoder.transform(self.train_processed['Personality'])
        
        # Apply feature selection and scaling
        X_full_scaled = self.scaler.transform(X_full)
        X_full_selected = self.feature_selector.transform(X_full_scaled)
        
        # Nested cross-validation
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        cv_scores = []
        
        for train_idx, test_idx in outer_cv.split(X_full_selected, y_full):
            X_train_cv, X_test_cv = X_full_selected[train_idx], X_full_selected[test_idx]
            y_train_cv, y_test_cv = y_full[train_idx], y_full[test_idx]
            
            # Train ensemble on this fold
            if hasattr(self, 'ensemble'):
                self.ensemble.fit(X_train_cv, y_train_cv)
                y_pred_cv = self.ensemble.predict(X_test_cv)
                accuracy = accuracy_score(y_test_cv, y_pred_cv)
                cv_scores.append(accuracy)
        
        print(f"Nested CV scores: {cv_scores}")
        print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
    def generate_optimized_predictions(self):
        """Generate optimized predictions for test set"""
        print("\n=== GENERATING OPTIMIZED PREDICTIONS ===")
        
        # Prepare test features
        X_test = self.test_processed[self.feature_columns]
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Make predictions with ensemble
        if hasattr(self, 'ensemble'):
            y_pred_proba = self.ensemble.predict_proba(X_test_selected)
        else:
            # Use best single model
            best_model_name = max(self.model_scores, key=self.model_scores.get)
            best_model = self.models[best_model_name]
            y_pred_proba = best_model.predict_proba(X_test_selected)
        
        # Convert probabilities to predictions
        y_pred = self.target_encoder.inverse_transform(y_pred_proba.argmax(axis=1))
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Personality': y_pred
        })
        
        # Save predictions
        submission.to_csv('optimized_submission.csv', index=False)
        print("Optimized predictions saved to optimized_submission.csv")
        
        # Display prediction distribution
        print("\nPrediction distribution:")
        print(submission['Personality'].value_counts())
        
        return submission
        
    def run_optimized_pipeline(self):
        """Run the complete optimized modeling pipeline"""
        print("=== OPTIMIZED PERSONALITY PREDICTION PIPELINE ===\n")
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Advanced preprocessing
        self.advanced_preprocess_data()
        
        # Prepare features with selection
        self.prepare_features_with_selection()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Create advanced ensemble
        ensemble_accuracy = self.create_advanced_ensemble()
        
        # Advanced cross-validation
        self.advanced_cross_validation()
        
        # Generate optimized predictions
        submission = self.generate_optimized_predictions()
        
        print("\n=== OPTIMIZED PIPELINE COMPLETED ===")
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        return submission

if __name__ == "__main__":
    # Create and run the optimized predictor
    predictor = OptimizedPersonalityPredictor()
    submission = predictor.run_optimized_pipeline() 