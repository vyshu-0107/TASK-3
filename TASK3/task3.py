"""
SALES PREDICTION USING PYTHON
A comprehensive machine learning project for sales forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class SalesPredictionModel:
    """
    A comprehensive class for sales prediction using machine learning
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
    def load_data(self, filepath):
        """Load sales data from CSV file"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"\nFirst few rows:\n{self.df.head()}")
            print(f"\nData Info:\n{self.df.info()}")
            print(f"\nBasic Statistics:\n{self.df.describe()}")
            return self.df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def generate_sample_data(self, n_samples=365):
        """Generate realistic sample sales data for demonstration"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
        
        # Create features
        day_of_week = dates.dayofweek
        month = dates.month
        quarter = dates.quarter
        
        # Simulate sales with trends and seasonality
        base_sales = 1000
        trend = np.arange(n_samples) * 0.5
        seasonality = 200 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
        noise = np.random.normal(0, 100, n_samples)
        
        sales = base_sales + trend + seasonality + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative sales
        
        # Additional features
        marketing_spend = np.random.uniform(100, 2000, n_samples)
        customers = np.random.randint(10, 200, n_samples)
        avg_transaction = np.random.uniform(20, 500, n_samples)
        
        self.df = pd.DataFrame({
            'Date': dates,
            'DayOfWeek': day_of_week,
            'Month': month,
            'Quarter': quarter,
            'MarketingSpend': marketing_spend,
            'Customers': customers,
            'AvgTransaction': avg_transaction,
            'Sales': sales
        })
        
        print(f"✓ Sample data generated: {self.df.shape[0]} records")
        print(f"\nSample data:\n{self.df.head(10)}")
        
        return self.df
    
    def preprocess_data(self, test_size=0.2):
        """Preprocess and prepare data for modeling"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Handle missing values
        print("\n✓ Handling missing values...")
        self.df = self.df.dropna()
        
        # Separate features and target
        X = self.df.drop(['Date', 'Sales'], axis=1)
        y = self.df['Sales']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Data split: Train={self.X_train.shape[0]}, Test={self.X_test.shape[0]}")
        print(f"✓ Features: {list(X.columns)}")
        print(f"✓ Target: Sales (Min: ${y.min():.2f}, Max: ${y.max():.2f})")
        
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("\n" + "="*50)
        print("TRAINING LINEAR REGRESSION")
        print("="*50)
        
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = model.predict(self.X_test_scaled)
        self.models['Linear Regression'] = model
        self._evaluate_model('Linear Regression', y_pred)
        
    def train_elastic_net(self, alpha=1.0, l1_ratio=0.5):
        """Train ElasticNet model"""
        print("\n" + "="*50)
        print("TRAINING ELASTICNET")
        print("="*50)
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state)
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = model.predict(self.X_test_scaled)
        self.models['ElasticNet'] = model
        self._evaluate_model('ElasticNet', y_pred)
        
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST")
        print("="*50)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        self.models['Random Forest'] = model
        self._evaluate_model('Random Forest', y_pred)
        
        # Feature importance
        self._plot_feature_importance(model, 'Random Forest')
        
    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1):
        """Train Gradient Boosting model"""
        print("\n" + "="*50)
        print("TRAINING GRADIENT BOOSTING")
        print("="*50)
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        self.models['Gradient Boosting'] = model
        self._evaluate_model('Gradient Boosting', y_pred)
        
        # Feature importance
        self._plot_feature_importance(model, 'Gradient Boosting')
        
    def _evaluate_model(self, model_name, y_pred):
        """Evaluate model performance"""
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        self.results[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Predictions': y_pred
        }
        
        print(f"\n✓ Model: {model_name}")
        print(f"  • R² Score:  {r2:.4f}")
        print(f"  • RMSE:      ${rmse:.2f}")
        print(f"  • MAE:       ${mae:.2f}")
        print(f"  • MAPE:      {mape:.2f}%")
        
    def _plot_feature_importance(self, model, model_name):
        """Plot feature importance"""
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
        plt.title(f'{model_name} - Feature Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        print(f"\n✓ Feature importance plot saved")
        plt.show()
        
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.drop('Predictions', axis=1)
        comparison_df = comparison_df.sort_values('R2', ascending=False)
        
        print("\n" + comparison_df.to_string())
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        comparison_df[['R2']].plot(kind='bar', ax=axes[0, 0], color='green')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim([0, 1])
        
        comparison_df[['RMSE']].plot(kind='bar', ax=axes[0, 1], color='red')
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE ($)')
        
        comparison_df[['MAE']].plot(kind='bar', ax=axes[1, 0], color='blue')
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_ylabel('MAE ($)')
        
        comparison_df[['MAPE']].plot(kind='bar', ax=axes[1, 1], color='orange')
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE (%)')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300)
        print("\n✓ Comparison plot saved")
        plt.show()
        
        return comparison_df
    
    def visualize_predictions(self):
        """Visualize actual vs predicted sales"""
        print("\n" + "="*50)
        print("VISUALIZING PREDICTIONS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Actual vs Predicted Sales - Top Models', fontsize=16)
        
        models_to_plot = list(self.results.keys())[:4]
        
        for idx, (ax, model_name) in enumerate(zip(axes.flat, models_to_plot)):
            y_pred = self.results[model_name]['Predictions']
            
            ax.scatter(self.y_test, y_pred, alpha=0.5, s=20)
            ax.plot([self.y_test.min(), self.y_test.max()], 
                   [self.y_test.min(), self.y_test.max()], 
                   'r--', lw=2)
            
            r2 = self.results[model_name]['R2']
            ax.set_xlabel('Actual Sales ($)')
            ax.set_ylabel('Predicted Sales ($)')
            ax.set_title(f'{model_name} (R² = {r2:.4f})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_visualization.png', dpi=300)
        print("✓ Prediction visualization saved")
        plt.show()
    
    def get_best_model(self):
        """Get the best performing model"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['R2'])
        best_model = self.models[best_model_name]
        
        print("\n" + "="*50)
        print("BEST MODEL")
        print("="*50)
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"  • R² Score: {self.results[best_model_name]['R2']:.4f}")
        print(f"  • RMSE: ${self.results[best_model_name]['RMSE']:.2f}")
        
        return best_model_name, best_model
    
    def predict_future_sales(self, future_data):
        """Predict sales for future data"""
        best_model_name, best_model = self.get_best_model()
        
        # Scale future data
        if isinstance(future_data, pd.DataFrame):
            future_data_scaled = self.scaler.transform(future_data)
        else:
            future_data_scaled = future_data
        
        predictions = best_model.predict(future_data_scaled)
        
        return predictions

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SALES PREDICTION PROJECT")
    print("="*50)
    
    # Initialize model
    predictor = SalesPredictionModel(random_state=42)
    
    # Generate sample data (or load your own with predictor.load_data())
    predictor.generate_sample_data(n_samples=365)
    
    # Preprocess data
    predictor.preprocess_data(test_size=0.2)
    
    # Train multiple models
    predictor.train_linear_regression()
    predictor.train_elastic_net(alpha=1.0, l1_ratio=0.5)
    predictor.train_random_forest(n_estimators=100, max_depth=10)
    predictor.train_gradient_boosting(n_estimators=100, learning_rate=0.1)
    
    # Compare models
    comparison_results = predictor.compare_models()
    
    # Visualize predictions
    predictor.visualize_predictions()
    
    # Get best model
    best_model_name, best_model = predictor.get_best_model()
    
    print("\n" + "="*50)
    print("PROJECT COMPLETED")
    print("="*50)
    print("\nGenerated Files:")
    print("  • model_comparison.png")
    print("  • predictions_visualization.png")
    print("  • feature_importance_*.png")