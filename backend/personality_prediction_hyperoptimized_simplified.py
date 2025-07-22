import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Advanced optimization
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Model interpretability
import shap

# Advanced statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seeds for reproducibility
np.random.seed(42)

class HyperOptimizedPersonalityPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.best_params = {}
        self.shap_values = {}
        self.ensemble_weights = {}
        
    def load_data(self):
        """Load and perform comprehensive data analysis"""
        print("Loading data...")
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        # Comprehensive data analysis
        print("\n=== COMPREHENSIVE DATA ANALYSIS ===")
        
        # Basic info
        print("\nTraining data info:")
        print(self.train_data.info())
        
        # Missing values analysis
        print("\nMissing values analysis:")
        missing_train = self.train_data.isnull().sum()
        missing_test = self.test_data.isnull().sum()
        print("Training data missing values:")
        print(missing_train[missing_train > 0])
        print("Test data missing values:")
        print(missing_test[missing_test > 0])
        
        # Target distribution with detailed analysis
        print("\nTarget distribution:")
        target_counts = self.train_data['Personality'].value_counts()
        print(target_counts)
        print(f"Class balance ratio: {target_counts.min() / target_counts.max():.3f}")
        
        # Statistical summary
        print("\nStatistical summary of numerical features:")
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        print(self.train_data[numerical_features].describe())
        
        # Outlier detection
        print("\nOutlier analysis:")
        for feature in numerical_features:
            Q1 = self.train_data[feature].quantile(0.25)
            Q3 = self.train_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.train_data[(self.train_data[feature] < Q1 - 1.5 * IQR) | 
                                     (self.train_data[feature] > Q3 + 1.5 * IQR)]
            print(f"{feature}: {len(outliers)} outliers ({len(outliers)/len(self.train_data)*100:.1f}%)")
        
    def advanced_data_visualization(self):
        """Create comprehensive visualizations for data understanding"""
        print("\n=== ADVANCED DATA VISUALIZATION ===")
        
        # Create comprehensive visualizations
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Target Distribution', 'Time Spent Alone', 'Social Event Attendance',
                          'Going Outside', 'Friends Circle Size', 'Post Frequency',
                          'Feature Correlation', 'Personality vs Features', 'Feature Distributions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Target distribution
        target_counts = self.train_data['Personality'].value_counts()
        fig.add_trace(
            go.Bar(x=target_counts.index, y=target_counts.values, name='Personality Distribution'),
            row=1, col=1
        )
        
        # Numerical features by personality
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        for i, feature in enumerate(numerical_features):
            row = (i // 2) + 1
            col = (i % 2) + 2
            
            introvert_data = self.train_data[self.train_data['Personality'] == 'Introvert'][feature]
            extrovert_data = self.train_data[self.train_data['Personality'] == 'Extrovert'][feature]
            
            fig.add_trace(
                go.Box(y=introvert_data, name='Introvert', boxpoints='outliers'),
                row=row, col=col
            )
            fig.add_trace(
                go.Box(y=extrovert_data, name='Extrovert', boxpoints='outliers'),
                row=row, col=col
            )
        
        # Correlation heatmap
        correlation_matrix = self.train_data[numerical_features].corr()
        fig.add_trace(
            go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.columns),
            row=3, col=1
        )
        
        # Feature importance visualization
        feature_importance_data = []
        for feature in numerical_features:
            introvert_mean = self.train_data[self.train_data['Personality'] == 'Introvert'][feature].mean()
            extrovert_mean = self.train_data[self.train_data['Personality'] == 'Extrovert'][feature].mean()
            feature_importance_data.append(abs(introvert_mean - extrovert_mean))
        
        fig.add_trace(
            go.Bar(x=numerical_features, y=feature_importance_data, name='Feature Importance'),
            row=3, col=2
        )
        
        # Distribution plots
        for i, feature in enumerate(numerical_features):
            fig.add_trace(
                go.Histogram(x=self.train_data[feature], name=feature, opacity=0.7),
                row=3, col=3
            )
        
        fig.update_layout(height=1200, title_text="Comprehensive Data Analysis")
        fig.write_html("comprehensive_data_analysis.html")
        print("Comprehensive data analysis saved to comprehensive_data_analysis.html")
        
    def ultra_advanced_preprocessing(self):
        """Ultra-advanced data preprocessing with sophisticated feature engineering"""
        print("\n=== ULTRA-ADVANCED DATA PREPROCESSING ===")
        
        # Create copies for preprocessing
        train_processed = self.train_data.copy()
        test_processed = self.test_data.copy()
        
        # Advanced categorical encoding with target encoding
        categorical_features = ['Stage_fear', 'Drained_after_socializing']
        
        for feature in categorical_features:
            # Create label encoders
            le = LabelEncoder()
            
            # Fill missing values with mode
            train_mode = train_processed[feature].mode()[0]
            train_processed[feature] = train_processed[feature].fillna(train_mode)
            test_processed[feature] = test_processed[feature].fillna(train_mode)
            
            # Fit on training data only
            le.fit(train_processed[feature].unique())
            
            # Transform both datasets
            train_processed[feature] = le.transform(train_processed[feature])
            test_processed[feature] = le.transform(test_processed[feature])
            
            self.label_encoders[feature] = le
            
        # Advanced numerical feature handling
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        # Multiple imputation strategies
        print("Applying advanced imputation strategies...")
        
        # KNN imputation with optimized parameters
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        # Fit on training data
        train_numerical = train_processed[numerical_features]
        train_imputed = knn_imputer.fit_transform(train_numerical)
        train_processed[numerical_features] = train_imputed
        
        # Transform test data
        test_numerical = test_processed[numerical_features]
        test_imputed = knn_imputer.transform(test_numerical)
        test_processed[numerical_features] = test_imputed
        
        # Ultra-advanced feature engineering
        print("Creating ultra-advanced engineered features...")
        
        # 1. Social Activity Composite Scores
        train_processed['social_activity_score'] = (
            train_processed['Social_event_attendance'] * 0.35 + 
            train_processed['Going_outside'] * 0.25 + 
            train_processed['Post_frequency'] * 0.20 +
            train_processed['Friends_circle_size'] * 0.20
        )
        test_processed['social_activity_score'] = (
            test_processed['Social_event_attendance'] * 0.35 + 
            test_processed['Going_outside'] * 0.25 + 
            test_processed['Post_frequency'] * 0.20 +
            test_processed['Friends_circle_size'] * 0.20
        )
        
        # 2. Introversion Indicators
        train_processed['introversion_indicators'] = (
            train_processed['Time_spent_Alone'] * 0.40 + 
            train_processed['Stage_fear'] * 0.30 + 
            train_processed['Drained_after_socializing'] * 0.30
        )
        test_processed['introversion_indicators'] = (
            test_processed['Time_spent_Alone'] * 0.40 + 
            test_processed['Stage_fear'] * 0.30 + 
            test_processed['Drained_after_socializing'] * 0.30
        )
        
        # 3. Social Efficiency Metrics
        train_processed['social_efficiency'] = train_processed['social_activity_score'] / (train_processed['Friends_circle_size'] + 1)
        test_processed['social_efficiency'] = test_processed['social_activity_score'] / (test_processed['Friends_circle_size'] + 1)
        
        # 4. Energy Balance Score
        train_processed['energy_balance'] = train_processed['social_activity_score'] - train_processed['introversion_indicators']
        test_processed['energy_balance'] = test_processed['social_activity_score'] - test_processed['introversion_indicators']
        
        # 5. Social Confidence Index
        train_processed['social_confidence'] = (1 - train_processed['Stage_fear']) * train_processed['social_activity_score']
        test_processed['social_confidence'] = (1 - test_processed['Stage_fear']) * test_processed['social_activity_score']
        
        # 6. Interaction Quality Score
        train_processed['interaction_quality'] = train_processed['social_activity_score'] * (1 - train_processed['Drained_after_socializing'])
        test_processed['interaction_quality'] = test_processed['social_activity_score'] * (1 - test_processed['Drained_after_socializing'])
        
        # 7. Social Network Efficiency
        train_processed['network_efficiency'] = train_processed['Friends_circle_size'] / (train_processed['social_activity_score'] + 1)
        test_processed['network_efficiency'] = test_processed['Friends_circle_size'] / (test_processed['social_activity_score'] + 1)
        
        # 8. Alone Time Ratio
        train_processed['alone_ratio'] = train_processed['Time_spent_Alone'] / (train_processed['Time_spent_Alone'] + train_processed['social_activity_score'] + 1)
        test_processed['alone_ratio'] = test_processed['Time_spent_Alone'] / (test_processed['Time_spent_Alone'] + test_processed['social_activity_score'] + 1)
        
        # 9. Advanced Interaction Features
        train_processed['fear_social_interaction'] = train_processed['Stage_fear'] * train_processed['Social_event_attendance']
        test_processed['fear_social_interaction'] = test_processed['Stage_fear'] * test_processed['Social_event_attendance']
        
        train_processed['drain_social_interaction'] = train_processed['Drained_after_socializing'] * train_processed['Social_event_attendance']
        test_processed['drain_social_interaction'] = test_processed['Drained_after_socializing'] * test_processed['Social_event_attendance']
        
        # 10. Polynomial Features for Key Interactions
        train_processed['alone_social_interaction'] = train_processed['Time_spent_Alone'] * train_processed['social_activity_score']
        test_processed['alone_social_interaction'] = test_processed['Time_spent_Alone'] * test_processed['social_activity_score']
        
        # 11. Ratio Features
        train_processed['friends_activity_ratio'] = train_processed['Friends_circle_size'] / (train_processed['social_activity_score'] + 1)
        test_processed['friends_activity_ratio'] = test_processed['Friends_circle_size'] / (test_processed['social_activity_score'] + 1)
        
        # 12. Categorical Interaction Features
        train_processed['fear_drain_interaction'] = train_processed['Stage_fear'] * train_processed['Drained_after_socializing']
        test_processed['fear_drain_interaction'] = test_processed['Stage_fear'] * test_processed['Drained_after_socializing']
        
        # 13. Advanced Statistical Features
        for feature in numerical_features:
            train_processed[f'{feature}_squared'] = train_processed[feature] ** 2
            test_processed[f'{feature}_squared'] = test_processed[feature] ** 2
            
            train_processed[f'{feature}_cubed'] = train_processed[feature] ** 3
            test_processed[f'{feature}_cubed'] = test_processed[feature] ** 3
        
        # 14. Cross-Feature Interactions
        interaction_pairs = [
            ('Time_spent_Alone', 'Social_event_attendance'),
            ('Friends_circle_size', 'Post_frequency'),
            ('Going_outside', 'Social_event_attendance'),
            ('Stage_fear', 'Friends_circle_size')
        ]
        
        for feat1, feat2 in interaction_pairs:
            train_processed[f'{feat1}_{feat2}_interaction'] = train_processed[feat1] * train_processed[feat2]
            test_processed[f'{feat1}_{feat2}_interaction'] = test_processed[feat1] * test_processed[feat2]
        
        # 15. Normalized Features
        for feature in numerical_features:
            train_processed[f'{feature}_normalized'] = (train_processed[feature] - train_processed[feature].mean()) / train_processed[feature].std()
            test_processed[f'{feature}_normalized'] = (test_processed[feature] - train_processed[feature].mean()) / test_processed[feature].std()
        
        # Store processed data
        self.train_processed = train_processed
        self.test_processed = test_processed
        
        print("Ultra-advanced data preprocessing completed!")
        print(f"Total features created: {len(train_processed.columns)}")
        
    def advanced_feature_selection(self):
        """Advanced feature selection with multiple methods"""
        print("\n=== ADVANCED FEATURE SELECTION ===")
        
        # Select comprehensive feature set
        feature_columns = [
            'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency', 'social_activity_score', 'introversion_indicators',
            'social_efficiency', 'energy_balance', 'social_confidence', 
            'interaction_quality', 'network_efficiency', 'alone_ratio',
            'fear_social_interaction', 'drain_social_interaction', 'alone_social_interaction',
            'friends_activity_ratio', 'fear_drain_interaction',
            'Time_spent_Alone_squared', 'Social_event_attendance_squared', 
            'Going_outside_squared', 'Friends_circle_size_squared', 'Post_frequency_squared',
            'Time_spent_Alone_cubed', 'Social_event_attendance_cubed',
            'Time_spent_Alone_Social_event_attendance_interaction',
            'Friends_circle_size_Post_frequency_interaction',
            'Going_outside_Social_event_attendance_interaction',
            'Stage_fear_Friends_circle_size_interaction',
            'Time_spent_Alone_normalized', 'Social_event_attendance_normalized',
            'Going_outside_normalized', 'Friends_circle_size_normalized', 'Post_frequency_normalized'
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
        
        # Advanced scaling with multiple scalers
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        
        # Test different scalers
        best_scaler = None
        best_score = 0
        
        for scaler_name, scaler in scalers.items():
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Test with a simple model
            test_model = RandomForestClassifier(n_estimators=50, random_state=42)
            test_model.fit(X_train_scaled, y_train)
            score = test_model.score(X_val_scaled, y_val)
            
            if score > best_score:
                best_score = score
                best_scaler = scaler_name
                self.scaler = scaler
                self.X_train_scaled = X_train_scaled
                self.X_val_scaled = X_val_scaled
        
        print(f"Best scaler: {best_scaler} (score: {best_score:.4f})")
        
        # Multiple feature selection methods
        print("Applying multiple feature selection methods...")
        
        # Method 1: SelectKBest with f_classif
        selector_kbest = SelectKBest(score_func=f_classif, k=20)
        X_train_kbest = selector_kbest.fit_transform(self.X_train_scaled, y_train)
        X_val_kbest = selector_kbest.transform(self.X_val_scaled)
        
        # Method 2: SelectKBest with mutual_info_classif
        selector_mutual = SelectKBest(score_func=mutual_info_classif, k=20)
        X_train_mutual = selector_mutual.fit_transform(self.X_train_scaled, y_train)
        X_val_mutual = selector_mutual.transform(self.X_val_scaled)
        
        # Method 3: SelectFromModel with RandomForest
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector_rf = SelectFromModel(rf_selector, prefit=False)
        X_train_rf = selector_rf.fit_transform(self.X_train_scaled, y_train)
        X_val_rf = selector_rf.transform(self.X_val_scaled)
        
        # Method 4: SelectFromModel with ExtraTrees
        et_selector = ExtraTreesClassifier(n_estimators=100, random_state=42)
        selector_et = SelectFromModel(et_selector, prefit=False)
        X_train_et = selector_et.fit_transform(self.X_train_scaled, y_train)
        X_val_et = selector_et.transform(self.X_val_scaled)
        
        # Evaluate all feature selection methods
        feature_selection_methods = {
            'SelectKBest_f': (selector_kbest, X_train_kbest, X_val_kbest),
            'SelectKBest_mutual': (selector_mutual, X_train_mutual, X_val_mutual),
            'SelectFromModel_RF': (selector_rf, X_train_rf, X_val_rf),
            'SelectFromModel_ET': (selector_et, X_train_et, X_val_et)
        }
        
        best_method = None
        best_score = 0
        
        for method_name, (selector, X_train_sel, X_val_sel) in feature_selection_methods.items():
            test_model = RandomForestClassifier(n_estimators=50, random_state=42)
            test_model.fit(X_train_sel, y_train)
            score = test_model.score(X_val_sel, y_val)
            
            print(f"{method_name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_method = method_name
                self.feature_selector = selector
                self.X_train = X_train_sel
                self.X_val = X_val_sel
        
        print(f"Best feature selection method: {best_method} (score: {best_score:.4f})")
        
        self.feature_columns = feature_columns
        self.target_encoder = target_encoder
        self.y_train = y_train
        self.y_val = y_val
        
        print(f"Final training set shape: {self.X_train.shape}")
        print(f"Final validation set shape: {self.X_val.shape}")
        
    def create_advanced_mlp_models(self):
        """Create and train advanced MLP models"""
        print("\n=== CREATING ADVANCED MLP MODELS ===")
        
        # Create multiple MLP architectures
        mlp_configs = [
            {
                'name': 'MLP_Deep',
                'hidden_layer_sizes': (256, 128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20
            },
            {
                'name': 'MLP_Wide',
                'hidden_layer_sizes': (512, 256),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20
            },
            {
                'name': 'MLP_Regularized',
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.01,
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20
            }
        ]
        
        mlp_scores = {}
        
        for config in mlp_configs:
            print(f"Training {config['name']}...")
            
            mlp = MLPClassifier(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                activation=config['activation'],
                solver=config['solver'],
                alpha=config['alpha'],
                learning_rate=config['learning_rate'],
                max_iter=config['max_iter'],
                early_stopping=config['early_stopping'],
                validation_fraction=config['validation_fraction'],
                n_iter_no_change=config['n_iter_no_change'],
                random_state=42
            )
            
            mlp.fit(self.X_train, self.y_train)
            y_pred_mlp = mlp.predict(self.X_val)
            mlp_accuracy = accuracy_score(self.y_val, y_pred_mlp)
            
            print(f"{config['name']} Accuracy: {mlp_accuracy:.4f}")
            
            self.models[config['name']] = mlp
            mlp_scores[config['name']] = mlp_accuracy
        
        return mlp_scores
        
    def bayesian_hyperparameter_optimization(self, skip_catboost=False):
        """Perform Bayesian hyperparameter optimization for multiple models"""
        print("\n=== BAYESIAN HYPERPARAMETER OPTIMIZATION ===")
        
        # Define search spaces for each model
        search_spaces = {
            'XGBoost': [
                Integer(50, 200, name='n_estimators'),
                Real(0.01, 0.3, name='learning_rate'),
                Integer(3, 10, name='max_depth'),
                Real(0.6, 1.0, name='subsample'),
                Real(0.6, 1.0, name='colsample_bytree'),
                Real(0.0, 1.0, name='reg_alpha'),
                Real(0.0, 1.0, name='reg_lambda')
            ],
            'LightGBM': [
                Integer(100, 500, name='n_estimators'),
                Real(0.01, 0.3, name='learning_rate'),
                Integer(3, 15, name='max_depth'),
                Real(0.6, 1.0, name='subsample'),
                Real(0.6, 1.0, name='colsample_bytree'),
                Real(0.0, 1.0, name='reg_alpha'),
                Real(0.0, 1.0, name='reg_lambda')
            ]
        }
        
        # Add CatBoost only if not skipped
        if not skip_catboost:
            search_spaces['CatBoost'] = [
                Integer(50, 150, name='n_estimators'),  # Reduced range
                Real(0.01, 0.2, name='learning_rate'),  # Reduced range
                Integer(3, 8, name='max_depth'),        # Reduced range
                Real(0.6, 1.0, name='subsample'),
                Real(0.6, 1.0, name='colsample_bytree'),
                Real(0.0, 1.0, name='reg_alpha'),
                Real(0.0, 1.0, name='reg_lambda')
            ]
        
        import signal
        import time
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Optimization timed out")
        
        for model_name, search_space in search_spaces.items():
            print(f"Optimizing {model_name}...")
            
            # Set timeout for CatBoost (shorter timeout)
            if model_name == 'CatBoost':
                timeout_seconds = 300  # 5 minutes for CatBoost
                n_calls = 20  # Reduced calls for CatBoost
            else:
                timeout_seconds = 600  # 10 minutes for others
                n_calls = 50
            
            @use_named_args(search_space)
            def objective(**params):
                try:
                    if model_name == 'XGBoost':
                        model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
                    elif model_name == 'LightGBM':
                        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                    elif model_name == 'CatBoost':
                        model = cb.CatBoostClassifier(**params, random_state=42, verbose=False, allow_writing_files=False)
                    
                    # Use fewer CV folds for CatBoost to speed up
                    cv_folds = 2 if model_name == 'CatBoost' else 3
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_folds, scoring='accuracy', n_jobs=1)
                    return -cv_scores.mean()  # Negative because we minimize
                except Exception as e:
                    print(f"Error in {model_name} optimization: {e}")
                    return 0.0  # Return worst possible score
            
            try:
                # Set up timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                
                # Run Bayesian optimization
                result = gp_minimize(
                    objective,
                    search_space,
                    n_calls=n_calls,
                    random_state=42,
                    n_jobs=1  # Use single job to avoid conflicts
                )
                
                # Cancel timeout
                signal.alarm(0)
                
                # Get best parameters
                best_params = dict(zip([param.name for param in search_space], result.x))
                
                # Train model with best parameters
                if model_name == 'XGBoost':
                    best_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
                elif model_name == 'LightGBM':
                    best_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
                elif model_name == 'CatBoost':
                    best_model = cb.CatBoostClassifier(**best_params, random_state=42, verbose=False, allow_writing_files=False)
                
                best_model.fit(self.X_train, self.y_train)
                y_pred = best_model.predict(self.X_val)
                accuracy = accuracy_score(self.y_val, y_pred)
                
                print(f"{model_name} - Best params: {best_params}")
                print(f"{model_name} - Best CV score: {-result.fun:.4f}")
                print(f"{model_name} - Validation accuracy: {accuracy:.4f}")
                
                self.models[model_name] = best_model
                self.best_params[model_name] = best_params
                
            except TimeoutError:
                print(f"{model_name} optimization timed out. Using default parameters.")
                # Use default parameters as fallback
                if model_name == 'XGBoost':
                    best_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                elif model_name == 'LightGBM':
                    best_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                elif model_name == 'CatBoost':
                    best_model = cb.CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False)
                
                best_model.fit(self.X_train, self.y_train)
                y_pred = best_model.predict(self.X_val)
                accuracy = accuracy_score(self.y_val, y_pred)
                
                print(f"{model_name} - Default params - Validation accuracy: {accuracy:.4f}")
                
                self.models[model_name] = best_model
                self.best_params[model_name] = {}
                
            except Exception as e:
                print(f"Error optimizing {model_name}: {e}")
                print(f"Skipping {model_name} optimization.")
                continue
        
    def create_advanced_ensemble(self):
        """Create sophisticated ensemble with multiple strategies"""
        print("\n=== CREATING ADVANCED ENSEMBLE ===")
        
        # Train additional models for ensemble
        additional_models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RidgeClassifier': RidgeClassifier(random_state=42),
            'SVC': SVC(probability=True, random_state=42),
            'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            'GaussianNB': GaussianNB()
        }
        
        # Train additional models
        for name, model in additional_models.items():
            if name in ['LogisticRegression', 'RidgeClassifier']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_val_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_val)
            
            accuracy = accuracy_score(self.y_val, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")
            self.models[name] = model
        
        # Create multiple ensemble strategies
        print("Creating multiple ensemble strategies...")
        
        # Strategy 1: Voting Classifier (Hard)
        voting_hard = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='hard'
        )
        voting_hard.fit(self.X_train, self.y_train)
        y_pred_hard = voting_hard.predict(self.X_val)
        hard_accuracy = accuracy_score(self.y_val, y_pred_hard)
        print(f"Hard Voting Accuracy: {hard_accuracy:.4f}")
        
        # Strategy 2: Voting Classifier (Soft) - exclude models without predict_proba
        soft_estimators = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                soft_estimators.append((name, model))
        
        if len(soft_estimators) > 1:
            voting_soft = VotingClassifier(
                estimators=soft_estimators,
                voting='soft'
            )
            voting_soft.fit(self.X_train, self.y_train)
            y_pred_soft = voting_soft.predict(self.X_val)
            soft_accuracy = accuracy_score(self.y_val, y_pred_soft)
            print(f"Soft Voting Accuracy: {soft_accuracy:.4f}")
        else:
            print("Skipping Soft Voting (insufficient models with predict_proba)")
            soft_accuracy = 0.0
        
        # Strategy 3: Stacking Classifier - use only models with predict_proba
        if len(soft_estimators) > 1:
            meta_classifier = LogisticRegression(random_state=42)
            stacking = StackingClassifier(
                estimators=soft_estimators,
                final_estimator=meta_classifier,
                cv=3,
                stack_method='predict_proba'
            )
            stacking.fit(self.X_train, self.y_train)
            y_pred_stacking = stacking.predict(self.X_val)
            stacking_accuracy = accuracy_score(self.y_val, y_pred_stacking)
            print(f"Stacking Accuracy: {stacking_accuracy:.4f}")
        else:
            print("Skipping Stacking (insufficient models with predict_proba)")
            stacking_accuracy = 0.0
        
        # Strategy 4: Weighted Average (optimize weights) - use only models with predict_proba
        if len(soft_estimators) > 1:
            def optimize_weights(weights):
                predictions = []
                for name, model in soft_estimators:
                    if name in ['LogisticRegression', 'RidgeClassifier']:
                        pred = model.predict_proba(self.X_val_scaled)[:, 1]
                    else:
                        pred = model.predict_proba(self.X_val)[:, 1]
                    predictions.append(pred)
                
                weighted_pred = np.zeros(len(self.y_val))
                for i, (pred, weight) in enumerate(zip(predictions, weights)):
                    weighted_pred += weight * pred
                
                y_pred_weighted = (weighted_pred > 0.5).astype(int)
                return -accuracy_score(self.y_val, y_pred_weighted)
        
                    # Optimize weights using Bayesian optimization
            n_models = len(soft_estimators)
            weight_space = [Real(0, 1) for _ in range(n_models)]
            
            result = gp_minimize(
                optimize_weights,
                weight_space,
                n_calls=100,
                random_state=42
            )
            
            # Normalize weights
            optimal_weights = np.array(result.x)
            optimal_weights = optimal_weights / optimal_weights.sum()
            
            # Calculate weighted ensemble predictions
            predictions = []
            for name, model in soft_estimators:
                if name in ['LogisticRegression', 'RidgeClassifier']:
                    pred = model.predict_proba(self.X_val_scaled)[:, 1]
                else:
                    pred = model.predict_proba(self.X_val)[:, 1]
                predictions.append(pred)
            
            weighted_pred = np.zeros(len(self.y_val))
            for i, (pred, weight) in enumerate(zip(predictions, optimal_weights)):
                weighted_pred += weight * pred
            
            y_pred_weighted = (weighted_pred > 0.5).astype(int)
            weighted_accuracy = accuracy_score(self.y_val, y_pred_weighted)
            print(f"Weighted Ensemble Accuracy: {weighted_accuracy:.4f}")
            print(f"Optimal weights: {dict(zip([name for name, _ in soft_estimators], optimal_weights))}")
        else:
            print("Skipping Weighted Ensemble (insufficient models with predict_proba)")
            weighted_accuracy = 0.0
            optimal_weights = []
        
        # Store the best ensemble
        ensemble_scores = {
            'Hard Voting': hard_accuracy,
            'Soft Voting': soft_accuracy,
            'Stacking': stacking_accuracy,
            'Weighted': weighted_accuracy
        }
        
        best_ensemble = max(ensemble_scores, key=ensemble_scores.get)
        print(f"Best ensemble method: {best_ensemble} ({ensemble_scores[best_ensemble]:.4f})")
        
        # Store ensemble models
        self.ensemble_models = {
            'Hard Voting': voting_hard
        }
        
        if len(soft_estimators) > 1:
            self.ensemble_models.update({
                'Soft Voting': voting_soft,
                'Stacking': stacking,
                'Weighted': (optimal_weights, dict(soft_estimators))
            })
        
        self.ensemble_weights = optimal_weights if len(soft_estimators) > 1 else []
        self.best_ensemble = best_ensemble
        
        return ensemble_scores
        
    def model_interpretability_analysis(self):
        """Perform comprehensive model interpretability analysis"""
        print("\n=== MODEL INTERPRETABILITY ANALYSIS ===")
        
        # SHAP analysis for tree-based models
        print("Performing SHAP analysis...")
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X_val)
                
                # Store SHAP values
                self.shap_values[name] = shap_values
                
                # Create SHAP summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, self.X_val, feature_names=self.feature_columns, show=False)
                plt.title(f'SHAP Summary Plot - {name}')
                plt.tight_layout()
                plt.savefig(f'shap_summary_{name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"SHAP analysis completed for {name}")
        
        # Feature importance comparison
        print("Creating feature importance comparison...")
        
        importance_df = pd.DataFrame()
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Use correct feature names based on the model type
                if name in ['LogisticRegression', 'RidgeClassifier']:
                    # These models use scaled features (original feature names)
                    feature_names = self.feature_columns
                else:
                    # Other models use selected features
                    feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
                
                # Ensure length matches
                if len(model.feature_importances_) == len(feature_names):
                    importance_df[name] = model.feature_importances_
                    if importance_df.empty:
                        importance_df.index = feature_names
            
            if not importance_df.empty:
                # Plot feature importance comparison
                plt.figure(figsize=(15, 10))
                importance_df.plot(kind='bar', figsize=(15, 8))
                plt.title('Feature Importance Comparison Across Models')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.xticks(rotation=45, ha='right')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save feature importance to CSV
                importance_df.to_csv('feature_importance_analysis.csv')
                print("Feature importance analysis saved to feature_importance_analysis.csv")
            else:
                print("No feature importance data available for plotting")
        
        # Model performance comparison
        print("Creating model performance comparison...")
        
        performance_metrics = {}
        
        for name, model in self.models.items():
            if name in ['LogisticRegression', 'RidgeClassifier']:
                y_pred = model.predict(self.X_val_scaled)
                # Handle models without predict_proba
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_val_scaled)[:, 1]
                else:
                    # For RidgeClassifier, use decision_function
                    y_pred_proba = model.decision_function(self.X_val_scaled)
            else:
                y_pred = model.predict(self.X_val)
                y_pred_proba = model.predict_proba(self.X_val)[:, 1]
            
            performance_metrics[name] = {
                'Accuracy': accuracy_score(self.y_val, y_pred),
                'Precision': precision_score(self.y_val, y_pred),
                'Recall': recall_score(self.y_val, y_pred),
                'F1-Score': f1_score(self.y_val, y_pred),
                'ROC-AUC': roc_auc_score(self.y_val, y_pred_proba)
            }
        
        # Create performance comparison plot
        metrics_df = pd.DataFrame(performance_metrics).T
        
        plt.figure(figsize=(12, 8))
        metrics_df.plot(kind='bar', figsize=(12, 8))
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save performance metrics
        metrics_df.to_csv('model_performance_metrics.csv')
        print("Model performance metrics saved to model_performance_metrics.csv")
        
        return performance_metrics
        
    def generate_hyperoptimized_predictions(self):
        """Generate hyperoptimized predictions for test set"""
        print("\n=== GENERATING HYPEROPTIMIZED PREDICTIONS ===")
        
        # Prepare test features
        X_test = self.test_processed[self.feature_columns]
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Make predictions with best ensemble
        if self.best_ensemble == 'Weighted':
            # Weighted ensemble predictions
            predictions = []
            for name, model in self.models.items():
                if name in ['LogisticRegression', 'RidgeClassifier']:
                    pred = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    pred = model.predict_proba(X_test_selected)[:, 1]
                predictions.append(pred)
            
            weighted_pred = np.zeros(len(X_test_selected))
            for i, (pred, weight) in enumerate(zip(predictions, self.ensemble_weights)):
                weighted_pred += weight * pred
            
            y_pred_proba = np.column_stack([1 - weighted_pred, weighted_pred])
        else:
            # Use the best ensemble model
            ensemble_model = self.ensemble_models[self.best_ensemble]
            y_pred_proba = ensemble_model.predict_proba(X_test_selected)
        
        # Convert probabilities to predictions
        y_pred = self.target_encoder.inverse_transform(y_pred_proba.argmax(axis=1))
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Personality': y_pred
        })
        
        # Save predictions
        submission.to_csv('hyperoptimized_submission.csv', index=False)
        print("Hyperoptimized predictions saved to hyperoptimized_submission.csv")
        
        # Display prediction distribution
        print("\nPrediction distribution:")
        print(submission['Personality'].value_counts())
        
        # Save prediction probabilities for analysis
        proba_df = pd.DataFrame(y_pred_proba, columns=['Introvert', 'Extrovert'])
        proba_df['id'] = self.test_data['id']
        proba_df['predicted_personality'] = y_pred
        proba_df.to_csv('prediction_probabilities.csv', index=False)
        print("Prediction probabilities saved to prediction_probabilities.csv")
        
        return submission
        
    def run_hyperoptimized_pipeline(self):
        """Run the complete hyperoptimized modeling pipeline"""
        print("=== HYPEROPTIMIZED PERSONALITY PREDICTION PIPELINE ===\n")
        
        # Load and analyze data
        self.load_data()
        self.advanced_data_visualization()
        
        # Ultra-advanced preprocessing
        self.ultra_advanced_preprocessing()
        
        # Advanced feature selection
        self.advanced_feature_selection()
        
        # Create advanced MLP models
        mlp_scores = self.create_advanced_mlp_models()
        
        # Bayesian hyperparameter optimization (skip CatBoost to avoid timeout issues)
        self.bayesian_hyperparameter_optimization(skip_catboost=True)
        
        # Create advanced ensemble
        ensemble_scores = self.create_advanced_ensemble()
        
        # Model interpretability analysis
        performance_metrics = self.model_interpretability_analysis()
        
        # Generate hyperoptimized predictions
        submission = self.generate_hyperoptimized_predictions()
        
        print("\n=== HYPEROPTIMIZED PIPELINE COMPLETED ===")
        print("MLP Model Scores:")
        for name, score in mlp_scores.items():
            print(f"  {name}: {score:.4f}")
        print(f"Best Ensemble Method: {self.best_ensemble}")
        print(f"Best Ensemble Score: {ensemble_scores[self.best_ensemble]:.4f}")
        
        # Print best model performance
        best_model = max(performance_metrics, key=lambda x: performance_metrics[x]['Accuracy'])
        print(f"Best Single Model: {best_model}")
        print(f"Best Model Accuracy: {performance_metrics[best_model]['Accuracy']:.4f}")
        
        return submission

    def run_hyperoptimized_pipeline_with_catboost(self):
        """Run the complete hyperoptimized modeling pipeline with CatBoost optimization"""
        print("=== HYPEROPTIMIZED PERSONALITY PREDICTION PIPELINE (WITH CATBOOST) ===\n")
        
        # Load and analyze data
        self.load_data()
        self.advanced_data_visualization()
        
        # Ultra-advanced preprocessing
        self.ultra_advanced_preprocessing()
        
        # Advanced feature selection
        self.advanced_feature_selection()
        
        # Create advanced MLP models
        mlp_scores = self.create_advanced_mlp_models()
        
        # Bayesian hyperparameter optimization (with CatBoost)
        self.bayesian_hyperparameter_optimization(skip_catboost=False)
        
        # Create advanced ensemble
        ensemble_scores = self.create_advanced_ensemble()
        
        # Model interpretability analysis
        performance_metrics = self.model_interpretability_analysis()
        
        # Generate hyperoptimized predictions
        submission = self.generate_hyperoptimized_predictions()
        
        print("\n=== HYPEROPTIMIZED PIPELINE COMPLETED ===")
        print("MLP Model Scores:")
        for name, score in mlp_scores.items():
            print(f"  {name}: {score:.4f}")
        print(f"Best Ensemble Method: {self.best_ensemble}")
        print(f"Best Ensemble Score: {ensemble_scores[self.best_ensemble]:.4f}")
        
        # Print best model performance
        best_model = max(performance_metrics, key=lambda x: performance_metrics[x]['Accuracy'])
        print(f"Best Single Model: {best_model}")
        print(f"Best Model Accuracy: {performance_metrics[best_model]['Accuracy']:.4f}")
        
        return submission

if __name__ == "__main__":
    # Create and run the hyperoptimized predictor
    predictor = HyperOptimizedPersonalityPredictor()
    submission = predictor.run_hyperoptimized_pipeline() 