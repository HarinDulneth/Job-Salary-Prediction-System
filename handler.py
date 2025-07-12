import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_salary import CareerPredictionModel
from model_title import JobTitlePredictionModel

class StudentPredictionAPI:
    """
    Enhanced API for student career prediction including salary and job title prediction
    """
    
    def __init__(self):
        """Initialize the prediction API"""
        self.data_loader = None
        self.feature_engineer = None
        self.salary_model = None
        self.job_title_model = None
        self.is_loaded = False
        self.comprehensive_df = None
        self.job_title_data = None
        
    def load_model(self, feature_engineer: FeatureEngineer, salary_model: CareerPredictionModel, 
                  job_title_model: JobTitlePredictionModel = None):
        """
        Load trained models and feature engineer
        
        Args:
            feature_engineer: Trained FeatureEngineer instance
            salary_model: Trained CareerPredictionModel instance
            job_title_model: Trained JobTitlePredictionModel instance (optional)
        """
        self.feature_engineer = feature_engineer
        self.salary_model = salary_model
        self.job_title_model = job_title_model
        self.is_loaded = True
        
    def setup_complete_system(self):
        """
        Setup the complete prediction system including job title prediction
        
        Returns:
            Dictionary with setup results
        """
        try:
            print("Setting up complete career prediction system...")
            
            # 1. Load and process data
            print("📊 Loading datasets...")
            self.data_loader = DataLoader()
            datasets = self.data_loader.load_all_datasets()
            self.comprehensive_df = self.data_loader.create_comprehensive_dataset()
            
            # 2. Feature engineering
            print("🔧 Engineering features...")
            self.feature_engineer = FeatureEngineer()
            engineered_df = self.feature_engineer.engineer_features(self.comprehensive_df)
            
            # 3. Setup salary prediction model
            print("💰 Setting up salary prediction...")
            features_df, target_series = self.feature_engineer.prepare_features_for_modeling(
                engineered_df, target_column='starting_salary_lkr'
            )
            
            selected_features_df, selected_feature_names = self.feature_engineer.select_best_features(
                features_df, target_series, k=20
            )
            self.feature_engineer.selected_feature_names = selected_feature_names
            
            self.salary_model = CareerPredictionModel(random_state=42)
            salary_results = self.salary_model.evaluate_models(selected_features_df, target_series)
            
            # 4. Setup job title prediction model
            print("🎯 Setting up job title prediction...")
            self.job_title_model = JobTitlePredictionModel(random_state=42)
            
            try:
                # Prepare job title data
                self.job_title_data = self.job_title_model.prepare_job_title_data(engineered_df)
                X_job, y_job = self.job_title_model.create_job_title_features(self.job_title_data)
                job_title_results = self.job_title_model.evaluate_job_title_models(X_job, y_job)
                
                job_title_setup = True
                job_title_message = f"Job title prediction ready with {len(self.job_title_model.label_encoder.classes_)} job categories"
                
            except Exception as e:
                print(f"Warning: Job title prediction setup failed: {str(e)}")
                job_title_setup = False
                job_title_message = f"Job title prediction not available: {str(e)}"
                self.job_title_model = None
            
            # 5. Mark system as loaded
            self.is_loaded = True
            
            results = {
                'success': True,
                'salary_model_ready': True,
                'job_title_model_ready': job_title_setup,
                'salary_best_model': self.salary_model.best_model_name,
                'job_title_best_model': self.job_title_model.best_model_name if job_title_setup else None,
                'total_students': len(self.comprehensive_df),
                'placed_students': len(self.job_title_data) if job_title_setup else 0,
                'message': f"System ready. Salary prediction: ✓, Job title prediction: {'✓' if job_title_setup else '✗'}",
                'job_title_message': job_title_message,
                'salary_model_results': salary_results.to_dict('records') if not salary_results.empty else [],
                'job_title_model_results': job_title_results.to_dict('records') if job_title_setup and not job_title_results.empty else []
            }
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"System setup failed: {str(e)}"
            }
    
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get the input schema for student prediction
        
        Returns:
            Dictionary describing required and optional input fields
        """
        schema = {
            'required_fields': [
                'student_id', 'gender', 'age_at_enrollment', 'province', 
                'z_score_AL', 'pathway', 'intake_year', 'current_semester'
            ],
            'optional_fields': [
                'current_gpa', 'completed_internships', 'internship_ratings',
                'total_internship_months', 'completed_projects', 'project_technologies',
                'certifications_earned', 'district', 'leadership_roles'
            ],
            'pathways': [
                'Artificial Intelligence', 'Data Science', 'Cyber Security',
                'Scientific Computing', 'Standard'
            ],
            'provinces': [
                'Western', 'Central', 'Southern', 'Northern', 'Eastern',
                'North Western', 'North Central', 'Uva', 'Sabaragamuwa'
            ]
        }
        
        if self.job_title_model and hasattr(self.job_title_model, 'label_encoder') and self.job_title_model.label_encoder:
            schema['available_job_titles'] = list(self.job_title_model.label_encoder.classes_)
        
        return schema
    
    def validate_input(self, student_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input student data
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        warnings = []
        schema = self.get_input_schema()
        
        # Check required fields
        for field in schema['required_fields']:
            if field not in student_data or student_data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate pathway
        if 'pathway' in student_data:
            if student_data['pathway'] not in schema['pathways']:
                errors.append(f"Invalid pathway. Must be one of: {schema['pathways']}")
        
        # Validate numeric ranges
        if 'z_score_AL' in student_data:
            try:
                z_score = float(student_data['z_score_AL'])
                if not (1.0 <= z_score <= 2.5):
                    errors.append("Z-score must be between 1.0 and 2.5")
            except (ValueError, TypeError):
                errors.append("Z-score must be a numeric value")
        
        if 'current_gpa' in student_data and student_data['current_gpa'] is not None:
            try:
                gpa = float(student_data['current_gpa'])
                if not (0.0 <= gpa <= 4.0):
                    errors.append("GPA must be between 0.0 and 4.0")
            except (ValueError, TypeError):
                errors.append("GPA must be a numeric value")
        
        # Warn about missing optional fields that improve accuracy
        if 'current_gpa' not in student_data or student_data['current_gpa'] is None:
            warnings.append("Current GPA not provided; using default value (3.0). Providing GPA improves prediction accuracy.")
        
        return len(errors) == 0, errors + warnings
    
    def create_student_dataframe(self, student_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a DataFrame from student input data
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            DataFrame with student data
        """
        # Create base row
        row_data = {}
        
        # Map input fields to expected column names
        field_mapping = {
            'student_id': 'student_id',
            'gender': 'gender', 
            'age_at_enrollment': 'age_at_enrollment',
            'province': 'province',
            'district': 'district',
            'z_score_AL': 'z_score_AL',
            'pathway': 'pathway',
            'intake_year': 'intake_year',
            'current_semester': 'current_semester',
            'current_gpa': 'cumulative_gpa'
        }
        
        for input_field, df_field in field_mapping.items():
            if input_field in student_data:
                row_data[df_field] = student_data[input_field]
        
        # Set default for cumulative_gpa if not provided
        if 'cumulative_gpa' not in row_data:
            row_data['cumulative_gpa'] = 3.0  # Default GPA (adjust as needed)
        
        # Handle experience data
        if 'completed_internships' in student_data:
            row_data['num_internships'] = student_data['completed_internships']
            
            if 'internship_ratings' in student_data and student_data['internship_ratings']:
                ratings = student_data['internship_ratings']
                row_data['avg_internship_rating'] = np.mean(ratings)
            
            if 'total_internship_months' in student_data:
                row_data['total_internship_months'] = student_data['total_internship_months']
        
        if 'completed_projects' in student_data:
            row_data['num_projects'] = student_data['completed_projects']
            
            if 'project_technologies' in student_data and student_data['project_technologies']:
                # Convert technologies list to string
                row_data['all_technologies'] = ', '.join(student_data['project_technologies'])
                row_data['technical_skills_count'] = len(student_data['project_technologies'])
        
        if 'certifications_earned' in student_data:
            row_data['num_certifications'] = student_data['certifications_earned']
        
        # Calculate derived fields
        current_year = datetime.now().year
        row_data['years_since_intake'] = current_year - student_data['intake_year']
        
        # Set default values for missing fields
        defaults = {
            'num_internships': 0,
            'avg_internship_rating': 0,
            'total_internship_months': 0,
            'num_projects': 0,
            'technical_skills_count': 0,
            'num_certifications': 0,
            'all_technologies': '',
            'district': student_data.get('district', 'Unknown')
        }
        
        for key, default_value in defaults.items():
            if key not in row_data:
                row_data[key] = default_value
        
        return pd.DataFrame([row_data])
    
    def predict_salary(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict salary for a student
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Dictionary with salary prediction and metadata
        """
        if not self.is_loaded:
            return {'error': 'Models not loaded. Call setup_complete_system() first.'}
        
        try:
            # Validate input
            is_valid, errors = self.validate_input(student_data)
            if not is_valid:
                return {'error': f'Input validation failed: {"; ".join(errors)}'}
            
            # Create DataFrame
            student_df = self.create_student_dataframe(student_data)
            
            # Apply feature engineering
            engineered_df = self.feature_engineer.engineer_features(student_df)
            
            # Prepare features for modeling
            features_df, _ = self.feature_engineer.prepare_features_for_modeling(
                engineered_df, target_column='starting_salary_lkr'
            )
            
            # Select features that were used in training
            if hasattr(self.feature_engineer, 'selected_feature_names'):
                available_features = [f for f in self.feature_engineer.selected_feature_names if f in features_df.columns]
                features_df = features_df[available_features]
            
            # Make prediction
            salary_prediction = self.salary_model.predict(features_df)[0]
            
            # Calculate confidence interval (simplified)
            base_uncertainty = 50000
            gpa = student_data.get('current_gpa', 3.0)
            experience_factor = len(student_data.get('project_technologies', [])) + student_data.get('completed_internships', 0)
            
            uncertainty = base_uncertainty * (1.5 - 0.1 * gpa - 0.05 * experience_factor)
            uncertainty = max(uncertainty, 25000)
            
            return {
                'predicted_salary': float(salary_prediction),
                'currency': 'LKR',
                'confidence_interval': {
                    'lower': float(salary_prediction - uncertainty),
                    'upper': float(salary_prediction + uncertainty)
                },
                'model_used': self.salary_model.best_model_name,
                'prediction_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def predict_job_title(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict job title for a student
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Dictionary with job title prediction and probabilities
        """
        if not self.is_loaded or not self.job_title_model:
            return {'error': 'Job title model not available'}
        
        try:
            # 1) Validate input
            is_valid, errors = self.validate_input(student_data)
            if not is_valid:
                return {'error': f'Input validation failed: {"; ".join(errors)}'}
            
            # 2) Build a one‐row DataFrame
            student_df = self.create_student_dataframe(student_data)
            
            # 3) Engineer features
            engineered_df = self.feature_engineer.engineer_features(student_df)
            
            # 4) Create the job‐title feature matrix
            X_features, _ = self.job_title_model.create_job_title_features(engineered_df)
            
            # 5) Convert any boolean columns (dtype=bool) to int
            bool_cols = (
                X_features.select_dtypes(include='boolean').columns.tolist() +
                X_features.select_dtypes(include='bool').columns.tolist()
            )
            for col in bool_cols:
                X_features[col] = X_features[col].astype(int)
            
            # 6) If any column somehow became a plain Python bool (scalar), convert it explicitly
            for col in X_features.columns:
                val = X_features.at[0, col]
                if isinstance(val, bool):
                    # Replace the entire column with a one‐element int()
                    X_features[col] = int(val)
            
            # 7) Run the trained pipeline
            predictions_encoded = self.job_title_model.predict(X_features)[0]
            try:
                probabilities = self.job_title_model.predict_proba(X_features)[0]
            except:
                probabilities = None
            
            # 8) Build top‐3 output with probabilities
            top_predictions = []
            if probabilities is not None:
                top_indices = np.argsort(probabilities)[-3:][::-1]
                for idx in top_indices:
                    job_title = self.job_title_model.label_encoder.inverse_transform([idx])[0]
                    prob = float(probabilities[idx])
                    top_predictions.append({
                        'job_title': job_title,
                        'probability': prob
                    })
            
            return {
                'predicted_job_title':
                    self.job_title_model.label_encoder.inverse_transform([predictions_encoded])[0],
                'top_predictions': top_predictions,
                'model_used': self.job_title_model.best_model_name,
                'prediction_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Job title prediction failed: {str(e)}'}

    
    def predict_complete_career_outcome(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict complete career outcome including salary and job title
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Dictionary with complete career predictions and insights
        """
        if not self.is_loaded:
            return {'error': 'Models not loaded. Call setup_complete_system() first.'}
        
        # Get salary prediction
        salary_result = self.predict_salary(student_data)
        
        # Get job title prediction
        job_title_result = self.predict_job_title(student_data)
        
        # Generate insights and recommendations
        insights = self._generate_insights(student_data, salary_result, job_title_result)
        recommendations = self._generate_recommendations(student_data, salary_result, job_title_result)
        
        # Compile complete result
        result = {
            'student_id': student_data.get('student_id', 'Unknown'),
            'salary_prediction': salary_result,
            'job_title_prediction': job_title_result,
            'insights': insights,
            'recommendations': recommendations,
            'student_profile': self._create_student_profile(student_data),
            'prediction_metadata': {
                'prediction_date': datetime.now().isoformat(),
                'system_version': '1.0',
                'models_used': {
                    'salary_model': self.salary_model.best_model_name if self.salary_model else None,
                    'job_title_model': self.job_title_model.best_model_name if self.job_title_model else None
                }
            }
        }
        
        return result
    
    def _generate_insights(self, student_data: Dict[str, Any], salary_result: Dict[str, Any], 
                          job_title_result: Dict[str, Any]) -> List[str]:
        """Generate insights based on predictions"""
        insights = []
        
        # Academic performance insights
        gpa = student_data.get('current_gpa')
        if gpa:
            if gpa >= 3.5:
                insights.append("Strong academic performance positions you well for competitive roles")
            elif gpa < 2.5:
                insights.append("Consider strategies to improve academic performance for better opportunities")
        
        # Experience insights
        internships = student_data.get('completed_internships', 0)
        projects = student_data.get('completed_projects', 0)
        
        if internships >= 2:
            insights.append("Multiple internships demonstrate strong industry experience")
        elif internships == 0 and student_data.get('current_semester', 0) >= 5:
            insights.append("Consider gaining internship experience to strengthen your profile")
        
        if projects >= 5:
            insights.append("Extensive project portfolio shows strong technical capabilities")
        
        # Pathway-specific insights
        pathway = student_data.get('pathway', '')
        high_demand_pathways = ['Artificial Intelligence', 'Data Science', 'Cyber Security']
        if pathway in high_demand_pathways:
            insights.append(f"{pathway} is a high-demand field with excellent career prospects")
        
        # Salary insights
        if 'predicted_salary' in salary_result:
            salary = salary_result['predicted_salary']
            if salary >= 80000:
                insights.append("Your predicted salary is above market average for entry-level positions")
            elif salary < 50000:
                insights.append("Focus on skill development to improve salary prospects")
        
        return insights
    
    def _generate_recommendations(self, student_data: Dict[str, Any], salary_result: Dict[str, Any], 
                                 job_title_result: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        semester = student_data.get('current_semester', 1)
        pathway = student_data.get('pathway', '')
        
        # Semester-based recommendations
        if semester <= 4:
            recommendations.append("Focus on building strong fundamentals and maintaining good grades")
            recommendations.append("Start exploring internship opportunities for upcoming semester breaks")
        elif semester <= 6:
            recommendations.append("Apply for internships to gain hands-on industry experience")
            recommendations.append("Work on substantial projects that showcase your technical skills")
        else:
            recommendations.append("Complete your capstone project with industry-relevant technologies")
            recommendations.append("Start networking and applying for full-time positions")
        
        # Skills and certification recommendations
        certifications = student_data.get('certifications_earned', 0)
        if certifications == 0:
            cert_recommendations = {
                'Artificial Intelligence': 'Consider AWS Machine Learning or Google Cloud ML certifications',
                'Data Science': 'Pursue Google Data Analytics or Microsoft Azure Data Scientist certifications',
                'Cyber Security': 'Look into CompTIA Security+ or CEH certifications',
                'Scientific Computing': 'Consider MATLAB or Python scientific computing certifications'
            }
            if pathway in cert_recommendations:
                recommendations.append(cert_recommendations[pathway])
        
        # Project recommendations
        projects = student_data.get('completed_projects', 0)
        if projects < 3:
            recommendations.append("Build more projects to demonstrate practical skills to employers")
        
        # Technology stack recommendations
        technologies = student_data.get('project_technologies', [])
        if len(technologies) < 5:
            tech_recommendations = {
                'Artificial Intelligence': 'Learn Python, TensorFlow, PyTorch, and cloud platforms',
                'Data Science': 'Master Python, R, SQL, Tableau, and statistical analysis tools',
                'Cyber Security': 'Gain expertise in security tools, network analysis, and ethical hacking',
                'Scientific Computing': 'Develop skills in MATLAB, Python, numerical computing libraries'
            }
            if pathway in tech_recommendations:
                recommendations.append(tech_recommendations[pathway])
        
        return recommendations
    
    def _create_student_profile(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive student profile summary"""
        gpa = student_data.get('current_gpa')
        
        # Calculate experience score
        internships = student_data.get('completed_internships', 0)
        projects = student_data.get('completed_projects', 0)
        certifications = student_data.get('certifications_earned', 0)
        technologies = len(student_data.get('project_technologies', []))
        
        experience_score = min(10, internships * 2 + projects * 0.5 + certifications * 1.5 + technologies * 0.3)
        
        # Categorize performance
        academic_level = "Not Available"
        if gpa:
            if gpa >= 3.7:
                academic_level = "Excellent"
            elif gpa >= 3.3:
                academic_level = "Good"
            elif gpa >= 2.7:
                academic_level = "Average"
            else:
                academic_level = "Below Average"
        
        experience_level = "Beginner"
        if experience_score >= 7:
            experience_level = "Advanced"
        elif experience_score >= 4:
            experience_level = "Intermediate"
        
        return {
            'pathway': student_data.get('pathway', 'Unknown'),
            'current_semester': f"{student_data.get('current_semester', 0)}/8",
            'academic_performance': academic_level,
            'experience_level': experience_level,
            'experience_score': round(experience_score, 1),
            'total_internships': internships,
            'total_projects': projects,
            'total_certifications': certifications,
            'technology_stack_size': technologies,
            'years_in_program': datetime.now().year - student_data.get('intake_year', datetime.now().year)
        }
    
    def get_career_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about career outcomes from the dataset
        
        Returns:
            Dictionary with career statistics
        """
        if not self.is_loaded or self.comprehensive_df is None:
            return {'error': 'System not loaded'}
        
        try:
            stats = {}
            
            # Salary statistics
            placed_students = self.comprehensive_df[self.comprehensive_df['placed'] == True]
            if not placed_students.empty:
                stats['salary_statistics'] = {
                    'average_salary': float(placed_students['starting_salary_lkr'].mean()),
                    'median_salary': float(placed_students['starting_salary_lkr'].median()),
                    'min_salary': float(placed_students['starting_salary_lkr'].min()),
                    'max_salary': float(placed_students['starting_salary_lkr'].max()),
                    'salary_by_pathway': placed_students.groupby('pathway')['starting_salary_lkr'].mean().to_dict()
                }
            
            # Job title statistics
            if 'job_title' in placed_students.columns:
                job_title_counts = placed_students['job_title'].value_counts()
                stats['job_title_statistics'] = {
                    'most_common_titles': job_title_counts.head(10).to_dict(),
                    'titles_by_pathway': placed_students.groupby('pathway')['job_title'].value_counts().to_dict()
                }
            
            # Placement statistics
            total_students = len(self.comprehensive_df)
            placed_count = len(placed_students)
            stats['placement_statistics'] = {
                'total_students': total_students,
                'placed_students': placed_count,
                'placement_rate': round((placed_count / total_students) * 100, 2),
                'placement_by_pathway': self.comprehensive_df.groupby('pathway')['placed'].mean().to_dict()
            }
            
            return stats
            
        except Exception as e:
            return {'error': f'Failed to calculate statistics: {str(e)}'}


# Example usage and testing
def test_prediction_system():
    """
    Test the complete prediction system with sample data
    """
    # Initialize the API
    api = StudentPredictionAPI()
    
    # Sample student data
    sample_student = {
        'student_id': 'CS2021001',
        'gender': 'Male',
        'age_at_enrollment': 19,
        'province': 'Western',
        'district': 'Colombo',
        'z_score_AL': 1.85,
        'pathway': 'Artificial Intelligence',
        'intake_year': 2021,
        'current_semester': 6,
        'current_gpa': 3.2,
        'completed_internships': 1,
        'internship_ratings': [4.5],
        'total_internship_months': 3,
        'completed_projects': 4,
        'project_technologies': ['Python', 'TensorFlow', 'React', 'SQL'],
        'certifications_earned': 1
    }
    
    print("Testing Career Prediction System")
    print("=" * 50)
    
    # Test input schema
    print("\n1. Input Schema:")
    schema = api.get_input_schema()
    print(f"Required fields: {schema['required_fields']}")
    print(f"Optional fields: {schema['optional_fields']}")
    
    # Test validation
    print("\n2. Input Validation:")
    is_valid, errors = api.validate_input(sample_student)
    print(f"Valid input: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    print("\n3. Sample Student Data:")
    for key, value in sample_student.items():
        print(f"  {key}: {value}")
    
    print("\nSystem ready for setup and prediction!")
    
    return api, sample_student


if __name__ == "__main__":
    # Test the system
    api, sample_data = test_prediction_system()
    
    print("\nTo use this system:")
    print("1. Call api.setup_complete_system() to initialize models")
    print("2. Use api.predict_complete_career_outcome(student_data) for predictions")
    print("3. Use api.get_career_statistics() for dataset insights")