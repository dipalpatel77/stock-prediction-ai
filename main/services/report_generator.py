"""
Report Generator Service
Handles report generation and formatting
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class ReportType(Enum):
    """Report type enumeration"""
    ANALYSIS = "analysis"
    PERFORMANCE = "performance"
    PREDICTION = "prediction"
    RISK = "risk"
    COMPREHENSIVE = "comprehensive"

class ReportFormat(Enum):
    """Report format enumeration"""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "excel"

@dataclass
class ReportSection:
    """Report section structure"""
    title: str
    content: str
    data: Dict[str, Any]
    charts: List[Dict[str, Any]]
    order: int

@dataclass
class Report:
    """Report structure"""
    report_id: str
    title: str
    report_type: ReportType
    format: ReportFormat
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    created_at: datetime
    file_path: str

class ReportGenerator:
    """Service for report generation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.reports_cache = {}
        self.cache_duration = timedelta(hours=2)
        
        # Report settings
        self.reports_dir = Path(self.config.get('reports_dir', 'reports'))
        self.reports_dir.mkdir(exist_ok=True)
        
        # Template settings
        self.templates_dir = Path(self.config.get('templates_dir', 'templates'))
        self.templates_dir.mkdir(exist_ok=True)
        
        # Report generation settings
        self.auto_generate = self.config.get('auto_generate', True)
        self.include_charts = self.config.get('include_charts', True)
        self.include_metadata = self.config.get('include_metadata', True)
        
        self.logger.info("Report Generator Service initialized")

    def generate_analysis_report(self, ticker: str, analysis_data: Dict[str, Any], 
                               format: ReportFormat = ReportFormat.HTML) -> str:
        """Generate analysis report"""
        try:
            report_id = f"analysis_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create report sections
            sections = self._create_analysis_sections(ticker, analysis_data)
            
            # Create report
            report = Report(
                report_id=report_id,
                title=f"Analysis Report for {ticker}",
                report_type=ReportType.ANALYSIS,
                format=format,
                sections=sections,
                metadata={
                    'ticker': ticker,
                    'generated_by': 'AI Stock Predictor',
                    'data_sources': analysis_data.get('data_sources', []),
                    'analysis_date': datetime.now().isoformat()
                },
                created_at=datetime.now(),
                file_path=""
            )
            
            # Generate report file
            file_path = self._generate_report_file(report)
            report.file_path = file_path
            
            self.logger.info(f"Generated analysis report: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating analysis report: {e}")
            return ""

    def generate_performance_report(self, ticker: str, performance_data: Dict[str, Any], 
                                  format: ReportFormat = ReportFormat.HTML) -> str:
        """Generate performance report"""
        try:
            report_id = f"performance_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create report sections
            sections = self._create_performance_sections(ticker, performance_data)
            
            # Create report
            report = Report(
                report_id=report_id,
                title=f"Performance Report for {ticker}",
                report_type=ReportType.PERFORMANCE,
                format=format,
                sections=sections,
                metadata={
                    'ticker': ticker,
                    'generated_by': 'AI Stock Predictor',
                    'performance_metrics': performance_data.get('metrics', {}),
                    'analysis_date': datetime.now().isoformat()
                },
                created_at=datetime.now(),
                file_path=""
            )
            
            # Generate report file
            file_path = self._generate_report_file(report)
            report.file_path = file_path
            
            self.logger.info(f"Generated performance report: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return ""

    def generate_prediction_report(self, ticker: str, prediction_data: Dict[str, Any], 
                                 format: ReportFormat = ReportFormat.HTML) -> str:
        """Generate prediction report"""
        try:
            report_id = f"prediction_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create report sections
            sections = self._create_prediction_sections(ticker, prediction_data)
            
            # Create report
            report = Report(
                report_id=report_id,
                title=f"Prediction Report for {ticker}",
                report_type=ReportType.PREDICTION,
                format=format,
                sections=sections,
                metadata={
                    'ticker': ticker,
                    'generated_by': 'AI Stock Predictor',
                    'prediction_horizon': prediction_data.get('horizon', '1 day'),
                    'confidence_level': prediction_data.get('confidence', 0.5),
                    'analysis_date': datetime.now().isoformat()
                },
                created_at=datetime.now(),
                file_path=""
            )
            
            # Generate report file
            file_path = self._generate_report_file(report)
            report.file_path = file_path
            
            self.logger.info(f"Generated prediction report: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating prediction report: {e}")
            return ""

    def generate_risk_report(self, ticker: str, risk_data: Dict[str, Any], 
                           format: ReportFormat = ReportFormat.HTML) -> str:
        """Generate risk report"""
        try:
            report_id = f"risk_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create report sections
            sections = self._create_risk_sections(ticker, risk_data)
            
            # Create report
            report = Report(
                report_id=report_id,
                title=f"Risk Report for {ticker}",
                report_type=ReportType.RISK,
                format=format,
                sections=sections,
                metadata={
                    'ticker': ticker,
                    'generated_by': 'AI Stock Predictor',
                    'risk_level': risk_data.get('risk_level', 'medium'),
                    'risk_factors': risk_data.get('risk_factors', []),
                    'analysis_date': datetime.now().isoformat()
                },
                created_at=datetime.now(),
                file_path=""
            )
            
            # Generate report file
            file_path = self._generate_report_file(report)
            report.file_path = file_path
            
            self.logger.info(f"Generated risk report: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return ""

    def generate_comprehensive_report(self, ticker: str, comprehensive_data: Dict[str, Any], 
                                    format: ReportFormat = ReportFormat.HTML) -> str:
        """Generate comprehensive report"""
        try:
            report_id = f"comprehensive_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create report sections
            sections = self._create_comprehensive_sections(ticker, comprehensive_data)
            
            # Create report
            report = Report(
                report_id=report_id,
                title=f"Comprehensive Report for {ticker}",
                report_type=ReportType.COMPREHENSIVE,
                format=format,
                sections=sections,
                metadata={
                    'ticker': ticker,
                    'generated_by': 'AI Stock Predictor',
                    'report_type': 'comprehensive',
                    'analysis_date': datetime.now().isoformat()
                },
                created_at=datetime.now(),
                file_path=""
            )
            
            # Generate report file
            file_path = self._generate_report_file(report)
            report.file_path = file_path
            
            self.logger.info(f"Generated comprehensive report: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return ""

    def _create_analysis_sections(self, ticker: str, analysis_data: Dict[str, Any]) -> List[ReportSection]:
        """Create analysis report sections"""
        try:
            sections = []
            
            # Executive Summary
            sections.append(ReportSection(
                title="Executive Summary",
                content=f"Analysis summary for {ticker}",
                data=analysis_data.get('summary', {}),
                charts=[],
                order=1
            ))
            
            # Technical Analysis
            if 'technical_analysis' in analysis_data:
                sections.append(ReportSection(
                    title="Technical Analysis",
                    content="Technical indicators and patterns",
                    data=analysis_data['technical_analysis'],
                    charts=self._create_technical_charts(analysis_data['technical_analysis']),
                    order=2
                ))
            
            # Fundamental Analysis
            if 'fundamental_analysis' in analysis_data:
                sections.append(ReportSection(
                    title="Fundamental Analysis",
                    content="Financial metrics and ratios",
                    data=analysis_data['fundamental_analysis'],
                    charts=self._create_fundamental_charts(analysis_data['fundamental_analysis']),
                    order=3
                ))
            
            # Market Sentiment
            if 'market_sentiment' in analysis_data:
                sections.append(ReportSection(
                    title="Market Sentiment",
                    content="Sentiment analysis and market indicators",
                    data=analysis_data['market_sentiment'],
                    charts=self._create_sentiment_charts(analysis_data['market_sentiment']),
                    order=4
                ))
            
            # Recommendations
            if 'recommendations' in analysis_data:
                sections.append(ReportSection(
                    title="Recommendations",
                    content="Investment recommendations and strategy",
                    data=analysis_data['recommendations'],
                    charts=[],
                    order=5
                ))
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error creating analysis sections: {e}")
            return []

    def _create_performance_sections(self, ticker: str, performance_data: Dict[str, Any]) -> List[ReportSection]:
        """Create performance report sections"""
        try:
            sections = []
            
            # Performance Overview
            sections.append(ReportSection(
                title="Performance Overview",
                content=f"Performance metrics for {ticker}",
                data=performance_data.get('overview', {}),
                charts=[],
                order=1
            ))
            
            # Model Performance
            if 'model_performance' in performance_data:
                sections.append(ReportSection(
                    title="Model Performance",
                    content="Machine learning model performance metrics",
                    data=performance_data['model_performance'],
                    charts=self._create_model_performance_charts(performance_data['model_performance']),
                    order=2
                ))
            
            # Historical Performance
            if 'historical_performance' in performance_data:
                sections.append(ReportSection(
                    title="Historical Performance",
                    content="Historical performance analysis",
                    data=performance_data['historical_performance'],
                    charts=self._create_historical_performance_charts(performance_data['historical_performance']),
                    order=3
                ))
            
            # Benchmark Comparison
            if 'benchmark_comparison' in performance_data:
                sections.append(ReportSection(
                    title="Benchmark Comparison",
                    content="Performance vs benchmark indices",
                    data=performance_data['benchmark_comparison'],
                    charts=self._create_benchmark_charts(performance_data['benchmark_comparison']),
                    order=4
                ))
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error creating performance sections: {e}")
            return []

    def _create_prediction_sections(self, ticker: str, prediction_data: Dict[str, Any]) -> List[ReportSection]:
        """Create prediction report sections"""
        try:
            sections = []
            
            # Prediction Summary
            sections.append(ReportSection(
                title="Prediction Summary",
                content=f"Price predictions for {ticker}",
                data=prediction_data.get('summary', {}),
                charts=[],
                order=1
            ))
            
            # Model Predictions
            if 'model_predictions' in prediction_data:
                sections.append(ReportSection(
                    title="Model Predictions",
                    content="Individual model predictions",
                    data=prediction_data['model_predictions'],
                    charts=self._create_prediction_charts(prediction_data['model_predictions']),
                    order=2
                ))
            
            # Ensemble Prediction
            if 'ensemble_prediction' in prediction_data:
                sections.append(ReportSection(
                    title="Ensemble Prediction",
                    content="Combined model prediction",
                    data=prediction_data['ensemble_prediction'],
                    charts=self._create_ensemble_charts(prediction_data['ensemble_prediction']),
                    order=3
                ))
            
            # Confidence Analysis
            if 'confidence_analysis' in prediction_data:
                sections.append(ReportSection(
                    title="Confidence Analysis",
                    content="Prediction confidence and uncertainty",
                    data=prediction_data['confidence_analysis'],
                    charts=self._create_confidence_charts(prediction_data['confidence_analysis']),
                    order=4
                ))
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error creating prediction sections: {e}")
            return []

    def _create_risk_sections(self, ticker: str, risk_data: Dict[str, Any]) -> List[ReportSection]:
        """Create risk report sections"""
        try:
            sections = []
            
            # Risk Overview
            sections.append(ReportSection(
                title="Risk Overview",
                content=f"Risk analysis for {ticker}",
                data=risk_data.get('overview', {}),
                charts=[],
                order=1
            ))
            
            # Market Risk
            if 'market_risk' in risk_data:
                sections.append(ReportSection(
                    title="Market Risk",
                    content="Market-related risk factors",
                    data=risk_data['market_risk'],
                    charts=self._create_market_risk_charts(risk_data['market_risk']),
                    order=2
                ))
            
            # Credit Risk
            if 'credit_risk' in risk_data:
                sections.append(ReportSection(
                    title="Credit Risk",
                    content="Credit and financial risk analysis",
                    data=risk_data['credit_risk'],
                    charts=self._create_credit_risk_charts(risk_data['credit_risk']),
                    order=3
                ))
            
            # Operational Risk
            if 'operational_risk' in risk_data:
                sections.append(ReportSection(
                    title="Operational Risk",
                    content="Operational and business risk factors",
                    data=risk_data['operational_risk'],
                    charts=self._create_operational_risk_charts(risk_data['operational_risk']),
                    order=4
                ))
            
            # Risk Mitigation
            if 'risk_mitigation' in risk_data:
                sections.append(ReportSection(
                    title="Risk Mitigation",
                    content="Risk mitigation strategies and recommendations",
                    data=risk_data['risk_mitigation'],
                    charts=[],
                    order=5
                ))
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error creating risk sections: {e}")
            return []

    def _create_comprehensive_sections(self, ticker: str, comprehensive_data: Dict[str, Any]) -> List[ReportSection]:
        """Create comprehensive report sections"""
        try:
            sections = []
            
            # Executive Summary
            sections.append(ReportSection(
                title="Executive Summary",
                content=f"Comprehensive analysis for {ticker}",
                data=comprehensive_data.get('summary', {}),
                charts=[],
                order=1
            ))
            
            # Market Analysis
            if 'market_analysis' in comprehensive_data:
                sections.append(ReportSection(
                    title="Market Analysis",
                    content="Market conditions and trends",
                    data=comprehensive_data['market_analysis'],
                    charts=self._create_market_analysis_charts(comprehensive_data['market_analysis']),
                    order=2
                ))
            
            # Technical Analysis
            if 'technical_analysis' in comprehensive_data:
                sections.append(ReportSection(
                    title="Technical Analysis",
                    content="Technical indicators and patterns",
                    data=comprehensive_data['technical_analysis'],
                    charts=self._create_technical_charts(comprehensive_data['technical_analysis']),
                    order=3
                ))
            
            # Fundamental Analysis
            if 'fundamental_analysis' in comprehensive_data:
                sections.append(ReportSection(
                    title="Fundamental Analysis",
                    content="Financial metrics and ratios",
                    data=comprehensive_data['fundamental_analysis'],
                    charts=self._create_fundamental_charts(comprehensive_data['fundamental_analysis']),
                    order=4
                ))
            
            # Risk Analysis
            if 'risk_analysis' in comprehensive_data:
                sections.append(ReportSection(
                    title="Risk Analysis",
                    content="Risk factors and mitigation strategies",
                    data=comprehensive_data['risk_analysis'],
                    charts=self._create_risk_analysis_charts(comprehensive_data['risk_analysis']),
                    order=5
                ))
            
            # Predictions
            if 'predictions' in comprehensive_data:
                sections.append(ReportSection(
                    title="Predictions",
                    content="Price predictions and forecasts",
                    data=comprehensive_data['predictions'],
                    charts=self._create_prediction_charts(comprehensive_data['predictions']),
                    order=6
                ))
            
            # Recommendations
            if 'recommendations' in comprehensive_data:
                sections.append(ReportSection(
                    title="Recommendations",
                    content="Investment recommendations and strategy",
                    data=comprehensive_data['recommendations'],
                    charts=[],
                    order=7
                ))
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive sections: {e}")
            return []

    def _create_technical_charts(self, technical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create technical analysis charts"""
        try:
            charts = []
            
            # Price chart
            if 'price_data' in technical_data:
                charts.append({
                    'type': 'line',
                    'title': 'Price Chart',
                    'data': technical_data['price_data'],
                    'x_axis': 'date',
                    'y_axis': 'price'
                })
            
            # Volume chart
            if 'volume_data' in technical_data:
                charts.append({
                    'type': 'bar',
                    'title': 'Volume Chart',
                    'data': technical_data['volume_data'],
                    'x_axis': 'date',
                    'y_axis': 'volume'
                })
            
            # Technical indicators
            if 'indicators' in technical_data:
                charts.append({
                    'type': 'multi_line',
                    'title': 'Technical Indicators',
                    'data': technical_data['indicators'],
                    'x_axis': 'date',
                    'y_axis': 'value'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating technical charts: {e}")
            return []

    def _create_fundamental_charts(self, fundamental_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fundamental analysis charts"""
        try:
            charts = []
            
            # Financial metrics
            if 'metrics' in fundamental_data:
                charts.append({
                    'type': 'bar',
                    'title': 'Financial Metrics',
                    'data': fundamental_data['metrics'],
                    'x_axis': 'metric',
                    'y_axis': 'value'
                })
            
            # Revenue growth
            if 'revenue_growth' in fundamental_data:
                charts.append({
                    'type': 'line',
                    'title': 'Revenue Growth',
                    'data': fundamental_data['revenue_growth'],
                    'x_axis': 'date',
                    'y_axis': 'growth_rate'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating fundamental charts: {e}")
            return []

    def _create_sentiment_charts(self, sentiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create sentiment analysis charts"""
        try:
            charts = []
            
            # Sentiment score
            if 'sentiment_score' in sentiment_data:
                charts.append({
                    'type': 'gauge',
                    'title': 'Sentiment Score',
                    'data': sentiment_data['sentiment_score'],
                    'min_value': 0,
                    'max_value': 1
                })
            
            # Sentiment over time
            if 'sentiment_timeline' in sentiment_data:
                charts.append({
                    'type': 'line',
                    'title': 'Sentiment Over Time',
                    'data': sentiment_data['sentiment_timeline'],
                    'x_axis': 'date',
                    'y_axis': 'sentiment'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment charts: {e}")
            return []

    def _create_model_performance_charts(self, model_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create model performance charts"""
        try:
            charts = []
            
            # Model comparison
            if 'model_comparison' in model_data:
                charts.append({
                    'type': 'bar',
                    'title': 'Model Performance Comparison',
                    'data': model_data['model_comparison'],
                    'x_axis': 'model',
                    'y_axis': 'score'
                })
            
            # Performance over time
            if 'performance_timeline' in model_data:
                charts.append({
                    'type': 'line',
                    'title': 'Performance Over Time',
                    'data': model_data['performance_timeline'],
                    'x_axis': 'date',
                    'y_axis': 'performance'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating model performance charts: {e}")
            return []

    def _create_historical_performance_charts(self, historical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create historical performance charts"""
        try:
            charts = []
            
            # Performance comparison
            if 'performance_comparison' in historical_data:
                charts.append({
                    'type': 'line',
                    'title': 'Performance Comparison',
                    'data': historical_data['performance_comparison'],
                    'x_axis': 'date',
                    'y_axis': 'performance'
                })
            
            # Returns distribution
            if 'returns_distribution' in historical_data:
                charts.append({
                    'type': 'histogram',
                    'title': 'Returns Distribution',
                    'data': historical_data['returns_distribution'],
                    'x_axis': 'returns',
                    'y_axis': 'frequency'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating historical performance charts: {e}")
            return []

    def _create_benchmark_charts(self, benchmark_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create benchmark comparison charts"""
        try:
            charts = []
            
            # Benchmark comparison
            if 'benchmark_comparison' in benchmark_data:
                charts.append({
                    'type': 'line',
                    'title': 'Benchmark Comparison',
                    'data': benchmark_data['benchmark_comparison'],
                    'x_axis': 'date',
                    'y_axis': 'performance'
                })
            
            # Relative performance
            if 'relative_performance' in benchmark_data:
                charts.append({
                    'type': 'bar',
                    'title': 'Relative Performance',
                    'data': benchmark_data['relative_performance'],
                    'x_axis': 'benchmark',
                    'y_axis': 'relative_performance'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating benchmark charts: {e}")
            return []

    def _create_prediction_charts(self, prediction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create prediction charts"""
        try:
            charts = []
            
            # Prediction timeline
            if 'prediction_timeline' in prediction_data:
                charts.append({
                    'type': 'line',
                    'title': 'Prediction Timeline',
                    'data': prediction_data['prediction_timeline'],
                    'x_axis': 'date',
                    'y_axis': 'price'
                })
            
            # Prediction confidence
            if 'prediction_confidence' in prediction_data:
                charts.append({
                    'type': 'area',
                    'title': 'Prediction Confidence',
                    'data': prediction_data['prediction_confidence'],
                    'x_axis': 'date',
                    'y_axis': 'confidence'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating prediction charts: {e}")
            return []

    def _create_ensemble_charts(self, ensemble_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create ensemble prediction charts"""
        try:
            charts = []
            
            # Ensemble prediction
            if 'ensemble_prediction' in ensemble_data:
                charts.append({
                    'type': 'line',
                    'title': 'Ensemble Prediction',
                    'data': ensemble_data['ensemble_prediction'],
                    'x_axis': 'date',
                    'y_axis': 'price'
                })
            
            # Model weights
            if 'model_weights' in ensemble_data:
                charts.append({
                    'type': 'pie',
                    'title': 'Model Weights',
                    'data': ensemble_data['model_weights']
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble charts: {e}")
            return []

    def _create_confidence_charts(self, confidence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create confidence analysis charts"""
        try:
            charts = []
            
            # Confidence distribution
            if 'confidence_distribution' in confidence_data:
                charts.append({
                    'type': 'histogram',
                    'title': 'Confidence Distribution',
                    'data': confidence_data['confidence_distribution'],
                    'x_axis': 'confidence',
                    'y_axis': 'frequency'
                })
            
            # Confidence over time
            if 'confidence_timeline' in confidence_data:
                charts.append({
                    'type': 'line',
                    'title': 'Confidence Over Time',
                    'data': confidence_data['confidence_timeline'],
                    'x_axis': 'date',
                    'y_axis': 'confidence'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating confidence charts: {e}")
            return []

    def _create_market_risk_charts(self, market_risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create market risk charts"""
        try:
            charts = []
            
            # Risk factors
            if 'risk_factors' in market_risk_data:
                charts.append({
                    'type': 'bar',
                    'title': 'Risk Factors',
                    'data': market_risk_data['risk_factors'],
                    'x_axis': 'factor',
                    'y_axis': 'risk_score'
                })
            
            # Risk over time
            if 'risk_timeline' in market_risk_data:
                charts.append({
                    'type': 'line',
                    'title': 'Risk Over Time',
                    'data': market_risk_data['risk_timeline'],
                    'x_axis': 'date',
                    'y_axis': 'risk_score'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating market risk charts: {e}")
            return []

    def _create_credit_risk_charts(self, credit_risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create credit risk charts"""
        try:
            charts = []
            
            # Credit metrics
            if 'credit_metrics' in credit_risk_data:
                charts.append({
                    'type': 'bar',
                    'title': 'Credit Metrics',
                    'data': credit_risk_data['credit_metrics'],
                    'x_axis': 'metric',
                    'y_axis': 'value'
                })
            
            # Credit score
            if 'credit_score' in credit_risk_data:
                charts.append({
                    'type': 'gauge',
                    'title': 'Credit Score',
                    'data': credit_risk_data['credit_score'],
                    'min_value': 0,
                    'max_value': 100
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating credit risk charts: {e}")
            return []

    def _create_operational_risk_charts(self, operational_risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create operational risk charts"""
        try:
            charts = []
            
            # Operational metrics
            if 'operational_metrics' in operational_risk_data:
                charts.append({
                    'type': 'bar',
                    'title': 'Operational Metrics',
                    'data': operational_risk_data['operational_metrics'],
                    'x_axis': 'metric',
                    'y_axis': 'value'
                })
            
            # Risk assessment
            if 'risk_assessment' in operational_risk_data:
                charts.append({
                    'type': 'radar',
                    'title': 'Risk Assessment',
                    'data': operational_risk_data['risk_assessment']
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating operational risk charts: {e}")
            return []

    def _create_market_analysis_charts(self, market_analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create market analysis charts"""
        try:
            charts = []
            
            # Market trends
            if 'market_trends' in market_analysis_data:
                charts.append({
                    'type': 'line',
                    'title': 'Market Trends',
                    'data': market_analysis_data['market_trends'],
                    'x_axis': 'date',
                    'y_axis': 'value'
                })
            
            # Market sentiment
            if 'market_sentiment' in market_analysis_data:
                charts.append({
                    'type': 'gauge',
                    'title': 'Market Sentiment',
                    'data': market_analysis_data['market_sentiment'],
                    'min_value': 0,
                    'max_value': 1
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating market analysis charts: {e}")
            return []

    def _create_risk_analysis_charts(self, risk_analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create risk analysis charts"""
        try:
            charts = []
            
            # Risk matrix
            if 'risk_matrix' in risk_analysis_data:
                charts.append({
                    'type': 'scatter',
                    'title': 'Risk Matrix',
                    'data': risk_analysis_data['risk_matrix'],
                    'x_axis': 'probability',
                    'y_axis': 'impact'
                })
            
            # Risk timeline
            if 'risk_timeline' in risk_analysis_data:
                charts.append({
                    'type': 'line',
                    'title': 'Risk Timeline',
                    'data': risk_analysis_data['risk_timeline'],
                    'x_axis': 'date',
                    'y_axis': 'risk_score'
                })
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating risk analysis charts: {e}")
            return []

    def _generate_report_file(self, report: Report) -> str:
        """Generate report file"""
        try:
            if report.format == ReportFormat.JSON:
                return self._generate_json_report(report)
            elif report.format == ReportFormat.HTML:
                return self._generate_html_report(report)
            elif report.format == ReportFormat.PDF:
                return self._generate_pdf_report(report)
            elif report.format == ReportFormat.CSV:
                return self._generate_csv_report(report)
            elif report.format == ReportFormat.EXCEL:
                return self._generate_excel_report(report)
            else:
                return self._generate_json_report(report)
                
        except Exception as e:
            self.logger.error(f"Error generating report file: {e}")
            return ""

    def _generate_json_report(self, report: Report) -> str:
        """Generate JSON report"""
        try:
            file_path = self.reports_dir / f"{report.report_id}.json"
            
            report_data = {
                'report_id': report.report_id,
                'title': report.title,
                'report_type': report.report_type.value,
                'format': report.format.value,
                'sections': [
                    {
                        'title': section.title,
                        'content': section.content,
                        'data': section.data,
                        'charts': section.charts,
                        'order': section.order
                    }
                    for section in report.sections
                ],
                'metadata': report.metadata,
                'created_at': report.created_at.isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {e}")
            return ""

    def _generate_html_report(self, report: Report) -> str:
        """Generate HTML report"""
        try:
            file_path = self.reports_dir / f"{report.report_id}.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report.title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                    .chart {{ margin: 20px 0; text-align: center; }}
                    .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{report.title}</h1>
                    <p>Generated on: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            for section in sorted(report.sections, key=lambda x: x.order):
                html_content += f"""
                <div class="section">
                    <h2>{section.title}</h2>
                    <p>{section.content}</p>
                """
                
                if section.data:
                    html_content += f"<div class='data'><pre>{json.dumps(section.data, indent=2, default=str)}</pre></div>"
                
                if section.charts:
                    for chart in section.charts:
                        html_content += f"<div class='chart'><h3>{chart['title']}</h3><p>Chart data: {json.dumps(chart['data'], indent=2)}</p></div>"
                
                html_content += "</div>"
            
            if report.metadata:
                html_content += f"""
                <div class="metadata">
                    <h3>Report Metadata</h3>
                    <pre>{json.dumps(report.metadata, indent=2, default=str)}</pre>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return ""

    def _generate_pdf_report(self, report: Report) -> str:
        """Generate PDF report"""
        try:
            # This would require additional libraries like reportlab or weasyprint
            # For now, generate HTML and suggest conversion
            html_file = self._generate_html_report(report)
            if html_file:
                self.logger.info(f"Generated HTML report: {html_file}. Convert to PDF using external tool.")
                return html_file
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            return ""

    def _generate_csv_report(self, report: Report) -> str:
        """Generate CSV report"""
        try:
            file_path = self.reports_dir / f"{report.report_id}.csv"
            
            # Create DataFrame from report data
            data = []
            for section in report.sections:
                if section.data:
                    for key, value in section.data.items():
                        data.append({
                            'section': section.title,
                            'key': key,
                            'value': str(value)
                        })
            
            if data:
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                return str(file_path)
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {e}")
            return ""

    def _generate_excel_report(self, report: Report) -> str:
        """Generate Excel report"""
        try:
            file_path = self.reports_dir / f"{report.report_id}.xlsx"
            
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Report ID': [report.report_id],
                    'Title': [report.title],
                    'Type': [report.report_type.value],
                    'Created At': [report.created_at.isoformat()]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Sections sheet
                sections_data = []
                for section in report.sections:
                    sections_data.append({
                        'Title': section.title,
                        'Content': section.content,
                        'Order': section.order
                    })
                pd.DataFrame(sections_data).to_excel(writer, sheet_name='Sections', index=False)
                
                # Data sheet
                if report.sections:
                    data = []
                    for section in report.sections:
                        if section.data:
                            for key, value in section.data.items():
                                data.append({
                                    'Section': section.title,
                                    'Key': key,
                                    'Value': str(value)
                                })
                    if data:
                        pd.DataFrame(data).to_excel(writer, sheet_name='Data', index=False)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error generating Excel report: {e}")
            return ""

    def get_report_summary(self) -> Dict[str, Any]:
        """Get report generation summary"""
        try:
            # Get report statistics
            report_files = list(self.reports_dir.glob("*.json"))
            total_reports = len(report_files)
            
            # Count by type
            type_counts = {}
            for report_file in report_files:
                try:
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                        report_type = report_data.get('report_type', 'unknown')
                        type_counts[report_type] = type_counts.get(report_type, 0) + 1
                except Exception as e:
                    self.logger.warning(f"Error reading report file {report_file}: {e}")
                    continue
            
            # Get recent reports
            recent_reports = []
            for report_file in sorted(report_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                try:
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                        recent_reports.append({
                            'report_id': report_data.get('report_id'),
                            'title': report_data.get('title'),
                            'type': report_data.get('report_type'),
                            'created_at': report_data.get('created_at')
                        })
                except Exception as e:
                    self.logger.warning(f"Error reading report file {report_file}: {e}")
                    continue
            
            return {
                'total_reports': total_reports,
                'type_counts': type_counts,
                'recent_reports': recent_reports,
                'summary_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting report summary: {e}")
            return {}
