# 🚀 Incremental Training Guide

## Overview

The AI Stock Predictor now includes comprehensive **Incremental Learning** capabilities that allow models to continuously improve and adapt to new market data without requiring full retraining. This feature implements model versioning, automatic updates, and performance tracking.

## 🎯 Key Features

### 1. **Incremental Learning Support**
- **Continuous Model Updates**: Models can be updated with new data without losing previous knowledge
- **Performance-Based Updates**: Only update when significant performance improvement is detected
- **Smart Training**: Uses existing model weights as starting point for new training

### 2. **Model Update Pipeline**
- **Automatic Update Detection**: Checks data freshness and model performance
- **Flexible Update Types**: Supports both incremental updates and full retraining
- **Batch Processing**: Update multiple models simultaneously

### 3. **Model Versioning**
- **Version History**: Track all model versions with metadata
- **Rollback Capability**: Revert to any previous version if needed
- **Performance Tracking**: Monitor performance improvements over time

## 📁 File Structure

```
partB_model/
├── incremental_learning.py          # Core incremental learning functionality
├── model_update_pipeline.py         # Model update pipeline
└── ...

incremental_training_cli.py          # Command-line interface
tests/unit/test_incremental_learning.py  # Unit tests
```

## 🛠️ Installation & Setup

The incremental training feature is automatically available with the existing project setup. No additional dependencies are required.

## 📖 Usage Guide

### Command Line Interface

The incremental training CLI provides easy access to all incremental learning features:

```bash
# Check if models need updates
python incremental_training_cli.py check-updates --ticker RELIANCE --mode simple

# Update a specific model
python incremental_training_cli.py update --ticker RELIANCE --mode simple

# Run automatic updates for multiple tickers
python incremental_training_cli.py auto-update --tickers RELIANCE AAPL MSFT

# View version history
python incremental_training_cli.py versions --ticker RELIANCE

# Rollback to a specific version
python incremental_training_cli.py rollback --ticker RELIANCE --version-id RELIANCE_simple_20241201_143022

# Show incremental learning status
python incremental_training_cli.py status

# Clean up old versions
python incremental_training_cli.py cleanup --ticker RELIANCE --max-versions 5
```

### Programmatic Usage

#### Basic Incremental Learning

```python
from partB_model.incremental_learning import (
    IncrementalLearningManager,
    IncrementalTrainingPipeline
)

# Create learning manager
learning_manager = IncrementalLearningManager()

# Create training pipeline
pipeline = IncrementalTrainingPipeline(learning_manager)

# Check if model needs update
update_check = pipeline.check_for_updates('RELIANCE', 'simple')
print(f"Needs update: {update_check['needs_update']}")

# Perform incremental update
if update_check['needs_update']:
    result = pipeline.perform_incremental_update('RELIANCE', 'simple', new_data)
    print(f"Update successful: {result['success']}")
```

#### Model Update Pipeline

```python
from partB_model.model_update_pipeline import ModelUpdatePipeline

# Create update pipeline
pipeline = ModelUpdatePipeline()

# Run automatic updates for multiple tickers
results = pipeline.run_automatic_updates(['RELIANCE', 'AAPL', 'MSFT'])

# Check results
for ticker, ticker_results in results.items():
    for mode, result in ticker_results.items():
        print(f"{ticker} ({mode}): {result['status']}")
```

#### Version Management

```python
from partB_model.incremental_learning import IncrementalLearningManager

# Create learning manager
manager = IncrementalLearningManager()

# Get version history
versions = manager.get_all_versions('RELIANCE', 'simple')
for version in versions:
    print(f"Version: {version.version_id}")
    print(f"Created: {version.created_at}")
    print(f"Performance: {version.performance_metrics}")

# Rollback to specific version
success = manager.rollback_to_version('RELIANCE', 'RELIANCE_simple_20241201_143022')
print(f"Rollback successful: {success}")
```

## 🔧 Configuration

### Update Configuration

The incremental training system uses configurable parameters:

```python
update_config = {
    'min_data_points': 100,        # Minimum new data points required
    'performance_threshold': 0.05,  # 5% improvement threshold
    'max_versions_kept': 10,        # Maximum versions to keep
    'backup_before_update': True,   # Backup before updating
    'validate_after_update': True   # Validate after updating
}
```

### Performance Thresholds

- **Default Threshold**: 5% performance improvement required for update
- **Configurable**: Can be adjusted based on requirements
- **Metrics**: Uses RMSE (Root Mean Square Error) for comparison

## 📊 Monitoring & Analytics

### Version History

Each model version includes comprehensive metadata:

```json
{
    "version_id": "RELIANCE_simple_20241201_143022",
    "created_at": "2024-12-01T14:30:22",
    "performance_metrics": {
        "rmse": 0.0234,
        "mae": 0.0187
    },
    "training_samples": 1500,
    "validation_samples": 300,
    "improvement": 0.12,
    "feature_columns": ["feature1", "feature2", ...]
}
```

### Performance Tracking

Track model performance over time:

```python
# Get performance history
versions = manager.get_all_versions('RELIANCE', 'simple')
performance_history = []

for version in versions:
    performance_history.append({
        'date': version.created_at,
        'rmse': version.performance_metrics['rmse'],
        'improvement': version.metadata.get('improvement', 0)
    })
```

## 🔄 Update Workflow

### 1. **Update Detection**
- Check data freshness (last update date)
- Evaluate current model performance
- Determine update type (incremental vs full retraining)

### 2. **Data Preparation**
- Fetch recent market data
- Preprocess data using existing pipeline
- Validate data quality and quantity

### 3. **Model Update**
- Load current model and scaler
- Prepare new training data
- Perform incremental training
- Evaluate performance improvement

### 4. **Version Management**
- Create backup of current model
- Save new model version
- Update version registry
- Schedule next update

### 5. **Validation**
- Test new model performance
- Compare with previous version
- Rollback if performance degrades

## 🧪 Testing

Run the incremental learning tests:

```bash
# Run unit tests
python -m pytest tests/unit/test_incremental_learning.py -v

# Run specific test
python -m pytest tests/unit/test_incremental_learning.py::TestIncrementalTrainingPipeline::test_prepare_incremental_data -v
```

## 📈 Best Practices

### 1. **Update Frequency**
- **Daily**: For high-frequency trading strategies
- **Weekly**: For medium-term strategies
- **Monthly**: For long-term strategies

### 2. **Data Quality**
- Ensure sufficient new data points (minimum 100)
- Validate data quality before updates
- Monitor for data anomalies

### 3. **Performance Monitoring**
- Track performance improvements over time
- Set appropriate performance thresholds
- Monitor for performance degradation

### 4. **Version Management**
- Keep reasonable number of versions (5-10)
- Regular cleanup of old versions
- Document significant changes

### 5. **Backup Strategy**
- Always backup before updates
- Test rollback procedures
- Maintain backup integrity

## 🚨 Troubleshooting

### Common Issues

#### 1. **Insufficient Data**
```
Error: Insufficient data for update: 50 < 100
```
**Solution**: Wait for more data or reduce `min_data_points` threshold

#### 2. **Performance Degradation**
```
Warning: No significant performance improvement
```
**Solution**: Model is performing well, no update needed

#### 3. **Version Conflicts**
```
Error: Version already exists
```
**Solution**: Use unique version IDs or cleanup old versions

#### 4. **Model Loading Errors**
```
Error: Could not load model
```
**Solution**: Check model file integrity, use rollback if needed

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

### Planned Features

1. **Adaptive Learning Rates**: Dynamic learning rate adjustment
2. **Ensemble Updates**: Update multiple model types simultaneously
3. **Real-time Updates**: Continuous learning with streaming data
4. **Performance Forecasting**: Predict model performance trends
5. **Automated Hyperparameter Tuning**: Optimize model parameters

### Integration Opportunities

1. **Web Dashboard**: Visual version management interface
2. **API Endpoints**: RESTful API for model updates
3. **Scheduled Jobs**: Automated update scheduling
4. **Alert System**: Performance degradation notifications

## 📚 API Reference

### Core Classes

#### `IncrementalLearningManager`
- `create_version_id(ticker, mode)`: Create unique version ID
- `get_latest_version(ticker, mode)`: Get most recent version
- `get_all_versions(ticker, mode)`: Get all versions
- `register_version(ticker, version)`: Register new version
- `rollback_to_version(ticker, version_id)`: Rollback to version

#### `IncrementalTrainingPipeline`
- `prepare_incremental_data(data, features, target)`: Prepare training data
- `evaluate_model_performance(model, X, y)`: Evaluate model performance
- `should_update_model(current, new)`: Determine if update needed
- `update_model_incrementally(ticker, mode, data)`: Perform incremental update

#### `ModelUpdatePipeline`
- `check_for_updates(ticker, mode)`: Check if updates needed
- `prepare_update_data(ticker, mode)`: Prepare update data
- `perform_incremental_update(ticker, mode, data)`: Perform update
- `run_automatic_updates(tickers, modes)`: Batch update multiple models

## 🤝 Contributing

When contributing to the incremental training feature:

1. **Follow Testing**: Add tests for new functionality
2. **Document Changes**: Update this guide for new features
3. **Performance Impact**: Ensure updates don't degrade performance
4. **Backward Compatibility**: Maintain compatibility with existing models

## 📞 Support

For issues or questions about incremental training:

1. Check the troubleshooting section
2. Review test cases for examples
3. Examine version history for similar issues
4. Create detailed bug reports with logs

---

**Note**: This incremental training system is designed to work seamlessly with the existing AI Stock Predictor infrastructure while providing powerful continuous learning capabilities.
