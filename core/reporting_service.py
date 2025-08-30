#!/usr/bin/env python3
"""
Reporting Service
Shared reporting and visualization service for all analysis tools
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import warnings

# Visualization imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Install plotly for interactive visualizations.")

warnings.filterwarnings('ignore')

# Import currency utilities
from .currency_utils import get_currency_symbol, format_price

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ReportingService:
    """Shared reporting and visualization service."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "csv").mkdir(exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b',
            'dark': '#e377c2'
        }
    
    def generate_summary_report(self, results: Dict, ticker: str, 
                              analysis_type: str = "analysis") -> Dict:
        """
        Generate a comprehensive summary report.
        
        Args:
            results: Analysis results dictionary
            ticker: Stock ticker
            analysis_type: Type of analysis performed
            
        Returns:
            Summary report dictionary
        """
        report = {
            'metadata': {
                'ticker': ticker,
                'analysis_type': analysis_type,
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'summary': {},
            'predictions': {},
            'metrics': {},
            'recommendations': {},
            'risk_assessment': {}
        }
        
        # Extract summary information
        if 'current_price' in results:
            report['summary']['current_price'] = results['current_price']
        
        if 'predictions' in results:
            report['predictions'] = self._extract_predictions(results['predictions'])
        
        if 'metrics' in results:
            report['metrics'] = self._extract_metrics(results['metrics'])
        
        if 'recommendations' in results:
            report['recommendations'] = results['recommendations']
        
        if 'risk_assessment' in results:
            report['risk_assessment'] = results['risk_assessment']
        
        # Generate recommendations if not present
        if 'recommendations' not in report or not report['recommendations']:
            report['recommendations'] = self._generate_recommendations(results)
        
        # Generate risk assessment if not present
        if 'risk_assessment' not in report or not report['risk_assessment']:
            report['risk_assessment'] = self._generate_risk_assessment(results)
        
        return report
    
    def create_visualizations(self, data: pd.DataFrame, results: Dict, 
                            ticker: str, save_charts: bool = True) -> Dict:
        """
        Create comprehensive visualizations.
        
        Args:
            data: Stock data DataFrame
            results: Analysis results
            ticker: Stock ticker
            save_charts: Whether to save charts to disk
            
        Returns:
            Dictionary with chart file paths
        """
        charts = {}
        
        try:
            # Price chart with predictions
            if 'predictions' in results:
                charts['price_prediction'] = self._create_price_prediction_chart(
                    data, results['predictions'], ticker, save_charts
                )
            
            # Technical indicators chart
            charts['technical_indicators'] = self._create_technical_indicators_chart(
                data, ticker, save_charts
            )
            
            # Model performance chart
            if 'metrics' in results:
                charts['model_performance'] = self._create_model_performance_chart(
                    results['metrics'], ticker, save_charts
                )
            
            # Feature importance chart
            if 'feature_importance' in results:
                charts['feature_importance'] = self._create_feature_importance_chart(
                    results['feature_importance'], ticker, save_charts
                )
            
            # Risk assessment chart
            if 'risk_assessment' in results:
                charts['risk_assessment'] = self._create_risk_assessment_chart(
                    results['risk_assessment'], ticker, save_charts
                )
            
            # Summary dashboard
            charts['summary_dashboard'] = self._create_summary_dashboard(
                data, results, ticker, save_charts
            )
            
        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")
        
        return charts
    
    def export_results(self, results: Dict, ticker: str, 
                      format: str = 'csv', include_charts: bool = True) -> Dict:
        """
        Export results to various formats.
        
        Args:
            results: Analysis results
            ticker: Stock ticker
            format: Export format ('csv', 'json', 'excel')
            include_charts: Whether to include charts
            
        Returns:
            Dictionary with exported file paths
        """
        exported_files = {}
        
        try:
            # Export predictions
            if 'predictions' in results:
                exported_files['predictions'] = self._export_predictions(
                    results['predictions'], ticker, format
                )
            
            # Export metrics
            if 'metrics' in results:
                exported_files['metrics'] = self._export_metrics(
                    results['metrics'], ticker, format
                )
            
            # Export summary report
            exported_files['summary'] = self._export_summary_report(
                results, ticker, format
            )
            
            # Export charts if requested
            if include_charts and 'data' in results:
                charts = self.create_visualizations(results['data'], results, ticker)
                exported_files['charts'] = charts
            
        except Exception as e:
            print(f"⚠️ Error exporting results: {e}")
        
        return exported_files
    
    def _create_price_prediction_chart(self, data: pd.DataFrame, predictions: Dict, 
                                     ticker: str, save_chart: bool) -> str:
        """Create price prediction chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot historical data
        ax.plot(data['Date'], data['Close'], label='Historical Price', 
                color=self.colors['primary'], linewidth=2)
        
        # Plot predictions if available
        if 'dates' in predictions and 'values' in predictions:
            ax.plot(predictions['dates'], predictions['values'], 
                   label='Predictions', color=self.colors['secondary'], 
                   linewidth=2, linestyle='--')
            
            # Add confidence intervals if available
            if 'confidence_intervals' in predictions:
                ci = predictions['confidence_intervals']
                ax.fill_between(predictions['dates'], ci['lower'], ci['upper'], 
                              alpha=0.3, color=self.colors['secondary'], 
                              label='Confidence Interval')
        
        ax.set_title(f'{ticker} Price Prediction', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        currency = get_currency_symbol(ticker)
        ax.set_ylabel(f'Price ({currency})', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_chart:
            chart_path = self.output_dir / "charts" / f"{ticker}_price_prediction.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(chart_path)
        else:
            plt.show()
            return ""
    
    def _create_technical_indicators_chart(self, data: pd.DataFrame, 
                                         ticker: str, save_chart: bool) -> str:
        """Create technical indicators chart."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # RSI
        if 'RSI' in data.columns:
            axes[0].plot(data['Date'], data['RSI'], color=self.colors['primary'])
            axes[0].axhline(y=70, color='red', linestyle='--', alpha=0.7)
            axes[0].axhline(y=30, color='green', linestyle='--', alpha=0.7)
            axes[0].set_title('RSI', fontweight='bold')
            axes[0].set_ylabel('RSI')
            axes[0].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            axes[1].plot(data['Date'], data['MACD'], label='MACD', 
                        color=self.colors['primary'])
            axes[1].plot(data['Date'], data['MACD_Signal'], label='Signal', 
                        color=self.colors['secondary'])
            axes[1].set_title('MACD', fontweight='bold')
            axes[1].set_ylabel('MACD')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            axes[2].plot(data['Date'], data['Close'], label='Price', 
                        color=self.colors['primary'])
            axes[2].plot(data['Date'], data['BB_Upper'], label='Upper BB', 
                        color=self.colors['warning'], alpha=0.7)
            axes[2].plot(data['Date'], data['BB_Lower'], label='Lower BB', 
                        color=self.colors['warning'], alpha=0.7)
            axes[2].fill_between(data['Date'], data['BB_Upper'], data['BB_Lower'], 
                               alpha=0.1, color=self.colors['warning'])
            axes[2].set_title('Bollinger Bands', fontweight='bold')
            currency = get_currency_symbol(ticker)
            axes[2].set_ylabel(f'Price ({currency})')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{ticker} Technical Indicators', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_chart:
            chart_path = self.output_dir / "charts" / f"{ticker}_technical_indicators.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(chart_path)
        else:
            plt.show()
            return ""
    
    def _create_model_performance_chart(self, metrics: Dict, ticker: str, 
                                      save_chart: bool) -> str:
        """Create model performance comparison chart."""
        if not metrics:
            return ""
        
        # Extract model names and performance metrics
        model_names = []
        rmse_scores = []
        r2_scores = []
        
        for model_name, model_metrics in metrics.items():
            if isinstance(model_metrics, dict):
                model_names.append(model_name)
                rmse_scores.append(model_metrics.get('rmse', 0))
                r2_scores.append(model_metrics.get('r2', 0))
        
        if not model_names:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        bars1 = ax1.bar(model_names, rmse_scores, color=self.colors['primary'])
        ax1.set_title('Model RMSE Comparison', fontweight='bold')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # R² comparison
        bars2 = ax2.bar(model_names, r2_scores, color=self.colors['secondary'])
        ax2.set_title('Model R² Comparison', fontweight='bold')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.suptitle(f'{ticker} Model Performance Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_chart:
            chart_path = self.output_dir / "charts" / f"{ticker}_model_performance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(chart_path)
        else:
            plt.show()
            return ""
    
    def _create_feature_importance_chart(self, feature_importance: pd.DataFrame, 
                                       ticker: str, save_chart: bool) -> str:
        """Create feature importance chart."""
        if feature_importance.empty:
            return ""
        
        # Get top 15 features
        top_features = feature_importance.head(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=self.colors['primary'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'{ticker} Feature Importance', fontsize=16, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_chart:
            chart_path = self.output_dir / "charts" / f"{ticker}_feature_importance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(chart_path)
        else:
            plt.show()
            return ""
    
    def _create_risk_assessment_chart(self, risk_assessment: Dict, 
                                    ticker: str, save_chart: bool) -> str:
        """Create risk assessment chart."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create radar chart for risk factors
        risk_factors = list(risk_assessment.keys())
        risk_values = list(risk_assessment.values())
        
        # Normalize values to 0-1 scale
        max_value = max(risk_values) if risk_values else 1
        normalized_values = [v/max_value for v in risk_values]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(risk_factors), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        normalized_values += normalized_values[:1]
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2, color=self.colors['primary'])
        ax.fill(angles, normalized_values, alpha=0.25, color=self.colors['primary'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(risk_factors)
        ax.set_ylim(0, 1)
        ax.set_title(f'{ticker} Risk Assessment', fontsize=16, fontweight='bold')
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_chart:
            chart_path = self.output_dir / "charts" / f"{ticker}_risk_assessment.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(chart_path)
        else:
            plt.show()
            return ""
    
    def _create_summary_dashboard(self, data: pd.DataFrame, results: Dict, 
                                ticker: str, save_chart: bool) -> str:
        """Create summary dashboard."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Price trend
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(data['Date'], data['Close'], color=self.colors['primary'])
        ax1.set_title(f'{ticker} Price Trend', fontweight='bold')
        currency = get_currency_symbol(ticker)
        ax1.set_ylabel(f'Price ({currency})')
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.bar(range(len(data)), data['Volume'], color=self.colors['secondary'], alpha=0.7)
        ax2.set_title('Volume', fontweight='bold')
        ax2.set_ylabel('Volume')
        
        # Key metrics
        ax3 = fig.add_subplot(gs[1, :])
        metrics_text = self._create_metrics_text(results)
        ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes, 
                fontsize=12, verticalalignment='center')
        ax3.set_title('Key Metrics', fontweight='bold')
        ax3.axis('off')
        
        # Recommendations
        ax4 = fig.add_subplot(gs[2, :])
        recommendations_text = self._create_recommendations_text(results)
        ax4.text(0.1, 0.5, recommendations_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center')
        ax4.set_title('Recommendations', fontweight='bold')
        ax4.axis('off')
        
        plt.suptitle(f'{ticker} Analysis Dashboard', fontsize=18, fontweight='bold')
        
        if save_chart:
            chart_path = self.output_dir / "charts" / f"{ticker}_dashboard.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(chart_path)
        else:
            plt.show()
            return ""
    
    def _extract_predictions(self, predictions: Dict) -> Dict:
        """Extract prediction information."""
        extracted = {}
        
        if isinstance(predictions, dict):
            for key, value in predictions.items():
                if isinstance(value, (list, np.ndarray)):
                    extracted[key] = value.tolist() if hasattr(value, 'tolist') else value
                else:
                    extracted[key] = value
        
        return extracted
    
    def _extract_metrics(self, metrics: Dict) -> Dict:
        """Extract metrics information."""
        extracted = {}
        
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, dict):
                    extracted[key] = {k: float(v) if isinstance(v, (int, float)) else v 
                                    for k, v in value.items()}
                else:
                    extracted[key] = float(value) if isinstance(value, (int, float)) else value
        
        return extracted
    
    def _generate_recommendations(self, results: Dict) -> Dict:
        """Generate trading recommendations."""
        recommendations = {
            'action': 'HOLD',
            'confidence': 'MEDIUM',
            'reasoning': 'Insufficient data for recommendation',
            'risk_level': 'MEDIUM'
        }
        
        # Add logic to generate recommendations based on results
        if 'predictions' in results and 'current_price' in results:
            current_price = results['current_price']
            predictions = results['predictions']
            
            if 'values' in predictions and len(predictions['values']) > 0:
                predicted_price = predictions['values'][0]
                price_change = (predicted_price - current_price) / current_price
                
                if price_change > 0.05:  # 5% increase
                    recommendations['action'] = 'BUY'
                    recommendations['confidence'] = 'HIGH' if abs(price_change) > 0.1 else 'MEDIUM'
                elif price_change < -0.05:  # 5% decrease
                    recommendations['action'] = 'SELL'
                    recommendations['confidence'] = 'HIGH' if abs(price_change) > 0.1 else 'MEDIUM'
                else:
                    recommendations['action'] = 'HOLD'
                    recommendations['confidence'] = 'MEDIUM'
                
                recommendations['reasoning'] = f"Predicted {price_change:.2%} price change"
        
        return recommendations
    
    def _generate_risk_assessment(self, results: Dict) -> Dict:
        """Generate risk assessment."""
        risk_assessment = {
            'volatility_risk': 'MEDIUM',
            'market_risk': 'MEDIUM',
            'liquidity_risk': 'LOW',
            'model_risk': 'MEDIUM',
            'overall_risk': 'MEDIUM'
        }
        
        # Add logic to assess risks based on results
        if 'metrics' in results:
            metrics = results['metrics']
            if 'rmse' in metrics:
                rmse = metrics['rmse']
                if rmse > 0.1:
                    risk_assessment['model_risk'] = 'HIGH'
                elif rmse < 0.05:
                    risk_assessment['model_risk'] = 'LOW'
        
        return risk_assessment
    
    def _export_predictions(self, predictions: Dict, ticker: str, format: str) -> str:
        """Export predictions to file."""
        if format == 'csv':
            file_path = self.output_dir / "csv" / f"{ticker}_predictions.csv"
            df = pd.DataFrame(predictions)
            df.to_csv(file_path, index=False)
        elif format == 'json':
            file_path = self.output_dir / "reports" / f"{ticker}_predictions.json"
            with open(file_path, 'w') as f:
                json.dump(predictions, f, indent=2)
        else:
            file_path = ""
        
        return str(file_path)
    
    def _export_metrics(self, metrics: Dict, ticker: str, format: str) -> str:
        """Export metrics to file."""
        if format == 'csv':
            file_path = self.output_dir / "csv" / f"{ticker}_metrics.csv"
            df = pd.DataFrame([metrics]).T.reset_index()
            df.columns = ['Metric', 'Value']
            df.to_csv(file_path, index=False)
        elif format == 'json':
            file_path = self.output_dir / "reports" / f"{ticker}_metrics.json"
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            file_path = ""
        
        return str(file_path)
    
    def _export_summary_report(self, results: Dict, ticker: str, format: str) -> str:
        """Export summary report to file."""
        report = self.generate_summary_report(results, ticker)
        
        if format == 'json':
            file_path = self.output_dir / "reports" / f"{ticker}_summary_report.json"
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            file_path = ""
        
        return str(file_path)
    
    def _create_metrics_text(self, results: Dict) -> str:
        """Create metrics text for dashboard."""
        text = ""
        
        if 'metrics' in results:
            metrics = results['metrics']
            for key, value in metrics.items():
                if isinstance(value, dict):
                    text += f"{key}:\n"
                    for k, v in value.items():
                        text += f"  {k}: {v:.4f}\n"
                else:
                    text += f"{key}: {value:.4f}\n"
        
        return text if text else "No metrics available"
    
    def _create_recommendations_text(self, results: Dict) -> str:
        """Create recommendations text for dashboard."""
        text = ""
        
        if 'recommendations' in results:
            recs = results['recommendations']
            for key, value in recs.items():
                text += f"{key}: {value}\n"
        
        return text if text else "No recommendations available"
