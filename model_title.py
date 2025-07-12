import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class JobTitlePredictionModel:
    """
    Machine learning model for predicting job titles based on student academic and project data
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the job title prediction model
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.label_encoder = None
        self.model_results = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_job_title_data(self, comprehensive_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset specifically for job title prediction
        
        Args:
            comprehensive_df: Comprehensive dataset from DataLoader
            
        Returns:
            DataFrame ready for job title prediction
        """
        print("Preparing job title prediction data...")
        
        # Only keep placed students with job titles
        df = comprehensive_df[
            (comprehensive_df['placed'] == True) & 
            (comprehensive_df['role_title'].notna())
        ].copy()
        
        if df.empty:
            raise ValueError("No placed students with job titles found in the dataset")
        
        print(f"Dataset prepared with {len(df)} placed students")
        print(f"Job titles distribution:")
        print(df['role_title'].value_counts())
        
        return df
    
    def create_job_title_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        print("Creating job title prediction features...")
    
        # Define features that are most relevant for job title prediction
        academic_features = [
            'cumulative_gpa', 'z_score_AL', 'pathway', 'years_since_intake'
        ]
        experience_features = [
            'num_internships', 'avg_internship_rating', 'total_internship_months',
            'num_projects', 'completed_projects_ratio', 'num_certifications',
            'pathway_specific_certs'
        ]
        skill_features = [
            'all_technologies', 'internship_roles', 'primary_skills', 
            'technical_skills_count', 'leadership_experience'
        ]
        demographic_features = ['gender', 'province', 'age_at_enrollment']
        
        # Select only available columns
        feature_columns = [col for col in (academic_features + experience_features + 
                                        skill_features + demographic_features) if col in df.columns]
        
        # Create feature matrix
        if not feature_columns:
            raise ValueError("No relevant features found in the input DataFrame")
        
        X = df[feature_columns].copy()
        
        # Convert boolean columns to integers
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)
        
        # Fill missing values
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            else:
                X[col] = X[col].fillna('')
        
        # Target variable
        y = df['role_title'] if 'role_title' in df.columns else pd.Series([None] * len(df))
        
        # Encode target variable if present
        if y.notna().any():
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = pd.Series([None] * len(X))
        
        print(f"Features created: {X.shape[1]} features, {len(y)} samples")
        if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
            print(f"Number of unique job titles: {len(self.label_encoder.classes_)}")
            print(f"Job title classes: {list(self.label_encoder.classes_)}")
        
        return X, pd.Series(y_encoded)
    
    def create_job_title_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline for job title prediction features
        
        Args:
            X: Feature matrix
            
        Returns:
            ColumnTransformer for preprocessing
        """
        # Identify column types
        numeric_features = []
        categorical_features = []
        text_features = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            elif col in ['all_technologies', 'internship_roles', 'primary_skills']:
                text_features.append(col)
            else:
                categorical_features.append(col)
        
        # Create preprocessing steps
        transformers = []
        
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                               categorical_features))
        
        # Handle text features
        for text_col in text_features:
            if text_col in X.columns:
                max_features = 50 if text_col == 'all_technologies' else 20
                transformers.append((f'text_{text_col}', 
                                   TfidfVectorizer(max_features=max_features, stop_words='english', lowercase=True), 
                                   text_col))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        return preprocessor
    
    def initialize_job_title_models(self) -> Dict[str, Any]:
        """
        Initialize classification models for job title prediction
        
        Returns:
            Dictionary of model name to model object mappings
        """
        self.models = {
            # Ensemble Models (usually perform well for classification)
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=20,
                min_samples_split=5,
                random_state=self.random_state, 
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100, 
                max_depth=20,
                random_state=self.random_state, 
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state, 
                eval_metric='mlogloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state, 
                verbose=-1
            ),
            
            # Linear Models
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                C=1.0
            ),
            
            # Support Vector Machine
            'svm_rbf': SVC(
                kernel='rbf', 
                C=1.0,
                random_state=self.random_state, 
                probability=True
            ),
            
            # Neural Network
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=500, 
                alpha=0.01,
                random_state=self.random_state
            )
        }
        
        return self.models
    
    def evaluate_job_title_models(self, X: pd.DataFrame, y: pd.Series, 
                                test_size: float = 0.2, cv_folds: int = 5) -> pd.DataFrame:
        """
        Evaluate all models for job title prediction
        
        Args:
            X: Feature matrix
            y: Target variable (encoded)
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with model evaluation results
        """
        print("Evaluating job title prediction models...")
        print("=" * 50)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Create preprocessor
        self.preprocessor = self.create_job_title_preprocessor(X)
        
        # Initialize models if not done
        if not self.models:
            self.initialize_job_title_models()
        
        results = []
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('classifier', model)
                ])
                
                # Cross-validation
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, cv=cv_folds, 
                    scoring='f1_weighted', n_jobs=-1
                )
                
                # Fit and predict on test set
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                test_accuracy = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Store results
                result = {
                    'model': name,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1
                }
                results.append(result)
                
                # Store model and pipeline for later use
                self.model_results[name] = {
                    'pipeline': pipeline,
                    'metrics': result,
                    'predictions': y_pred,
                    'test_labels': y_test
                }
                
                print(f"  CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"  Test Accuracy: {test_accuracy:.3f}")
                print(f"  Test F1: {test_f1:.3f}")
                
            except Exception as e:
                print(f"  Error evaluating {name}: {str(e)}")
                continue
        
        # Convert to DataFrame and sort by test F1
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('test_f1', ascending=False)
            
            # Identify best model
            self.best_model_name = results_df.iloc[0]['model']
            self.best_model = self.model_results[self.best_model_name]['pipeline']
            self.is_trained = True
            
            print(f"\nBest Model: {self.best_model_name}")
            print("=" * 50)
        
        return results_df
    
    def get_job_title_classification_report(self, model_name: str = None) -> str:
        """
        Get detailed classification report for job title prediction
        
        Args:
            model_name: Name of model (uses best model if None)
            
        Returns:
            Classification report as string
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.model_results:
            return "Model not found"
        
        y_test = self.model_results[model_name]['test_labels']
        y_pred = self.model_results[model_name]['predictions']
        
        # Convert back to original labels
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        report = classification_report(y_test_labels, y_pred_labels)
        return report
    
    def get_job_title_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance for job title prediction
        
        Args:
            model_name: Name of model (uses best model if None)
            
        Returns:
            DataFrame with feature importance scores
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.model_results:
            return pd.DataFrame()
        
        pipeline = self.model_results[model_name]['pipeline']
        model = pipeline.named_steps['classifier']
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            try:
                feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                importance = model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                return importance_df.head(20)  # Top 20 features
            except Exception as e:
                print(f"Could not extract feature names: {e}")
                return pd.DataFrame()
        
        elif hasattr(model, 'coef_'):
            # Linear models
            try:
                feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                if len(model.coef_.shape) > 1:
                    importance = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importance = np.abs(model.coef_)
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                return importance_df.head(20)  # Top 20 features
            except Exception as e:
                print(f"Could not extract feature names: {e}")
                return pd.DataFrame()
        else:
            print("Model doesn't support feature importance")
            return pd.DataFrame()
    
    def predict_job_title(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make job title predictions using the best model
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        if not self.is_trained or self.best_model is None:
            raise ValueError("No trained model found. Train a model first.")
        
        predictions_encoded = self.best_model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        try:
            probabilities = self.best_model.predict_proba(X)
        except:
            probabilities = None
        
        return predictions, probabilities
    
    def get_job_title_probabilities_with_names(self, X: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Get job title predictions with probability scores for each class
        
        Args:
            X: Feature matrix
            
        Returns:
            List of dictionaries with job title probabilities
        """
        if not self.is_trained or self.best_model is None:
            raise ValueError("No trained model found. Train a model first.")
        
        probabilities = self.best_model.predict_proba(X)
        class_names = self.label_encoder.classes_
        
        results = []
        for prob_row in probabilities:
            prob_dict = {}
            for i, prob in enumerate(prob_row):
                prob_dict[class_names[i]] = float(prob)
            # Sort by probability descending
            prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            results.append(prob_dict)
        
        return results
    
    def tune_job_title_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                     model_name: str = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for job title prediction model
        
        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name of model to tune (uses best model if None)
            
        Returns:
            Dictionary with tuning results
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name is None:
            raise ValueError("No best model found. Run evaluate_job_title_models first.")
        
        print(f"Tuning hyperparameters for {model_name}...")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            },
            'logistic_regression': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return {}
        
        # Create pipeline
        base_model = self.models[model_name]
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', base_model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, param_grids[model_name],
            cv=3, scoring='f1_weighted',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Best CV F1 Score: {results['best_score']:.3f}")
        
        return results