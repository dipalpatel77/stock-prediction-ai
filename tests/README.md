# 🧪 AI Stock Predictor Test Suite

This directory contains a comprehensive test suite for the AI Stock Predictor system, organized into different categories for better management and execution.

## 📁 Directory Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_data_loader.py
│   ├── test_model_builder.py
│   └── test_strategy_components.py
├── integration/             # Integration tests for system components
│   ├── test_system_improvements.py
│   └── test_full_pipeline.py
├── performance/             # Performance and load tests
│   └── test_performance.py
├── utils/                   # Test utilities and helpers
│   └── test_helpers.py
├── data/                    # Test data files
├── results/                 # Test results and reports
│   └── comprehensive_test_results.md
├── test_config.json         # Test configuration
├── run_all_tests.py         # Main test runner
├── run_tests.bat           # Windows batch file for running tests
└── README.md               # This file
```

## 🚀 Quick Start

### Running All Tests
```bash
# Run all tests
python tests/run_all_tests.py

# Run with verbose output
python tests/run_all_tests.py --verbose

# Run specific test type
python tests/run_all_tests.py --type unit
python tests/run_all_tests.py --type integration
python tests/run_all_tests.py --type performance
```

### Running Specific Tests
```bash
# Run a specific test file
python tests/run_all_tests.py --file tests.unit.test_data_loader

# Run with custom configuration
python tests/run_all_tests.py --config tests/test_config.json
```

### Windows Batch File
```cmd
# Run tests using the batch file
tests\run_tests.bat
```

## 📋 Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Coverage**: Data loaders, model builders, strategy components
- **Execution**: Fast, focused on specific functionality
- **Dependencies**: Minimal, mostly mocked

### Integration Tests (`tests/integration/`)
- **Purpose**: Test system components working together
- **Coverage**: Full pipeline, end-to-end workflows
- **Execution**: Medium speed, tests component interactions
- **Dependencies**: May require actual data or models

### Performance Tests (`tests/performance/`)
- **Purpose**: Test system performance and load handling
- **Coverage**: Timing, memory usage, concurrent processing
- **Execution**: Slower, focuses on performance metrics
- **Dependencies**: May require significant resources

## 🛠️ Test Utilities

### Test Helpers (`tests/utils/test_helpers.py`)
- **TestDataGenerator**: Generate sample data for testing
- **TestEnvironmentManager**: Manage test environment and cleanup
- **TestAssertions**: Custom assertions for common test patterns
- **MockDataProvider**: Mock data for isolated testing

### Configuration (`tests/test_config.json`)
- Test mode settings
- Performance thresholds
- Test data parameters
- Output directory configuration

## 📊 Test Results

Test results are automatically saved to `tests/results/` with:
- Detailed test reports
- Performance metrics
- Error logs
- Timestamped results

## 🔧 Customization

### Adding New Tests
1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Use provided test utilities and helpers
4. Add to appropriate test category

### Test Configuration
Modify `tests/test_config.json` to adjust:
- Performance thresholds
- Test data parameters
- Logging levels
- Output settings

### Test Utilities
Extend `tests/utils/test_helpers.py` for:
- Custom test data generators
- Additional assertions
- Environment management
- Mock data providers

## 📈 Performance Benchmarks

The test suite includes performance benchmarks for:
- **Data Loading**: < 30 seconds
- **Feature Preparation**: < 10 seconds
- **Model Training**: < 60 seconds
- **Prediction Generation**: < 5 seconds
- **Memory Usage**: < 500 MB

## 🐛 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure parent directory is in Python path
2. **Missing Dependencies**: Install required packages from `requirements.txt`
3. **Memory Issues**: Reduce test data size in configuration
4. **Timeout Errors**: Increase timeout in configuration

### Debug Mode
```bash
# Run with debug logging
python tests/run_all_tests.py --verbose --config tests/test_config.json
```

## 📝 Best Practices

1. **Test Isolation**: Each test should be independent
2. **Mock External Dependencies**: Use mocks for API calls and file I/O
3. **Cleanup**: Always clean up test resources
4. **Descriptive Names**: Use clear, descriptive test names
5. **Documentation**: Document complex test scenarios
6. **Performance**: Monitor test execution time and memory usage

## 🔄 Continuous Integration

The test suite is designed to work with CI/CD pipelines:
- Exit codes for automated testing
- Structured output for reporting
- Configurable timeouts and thresholds
- Detailed logging for debugging

## 📚 Additional Resources

- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)
- [Test-driven development guide](https://en.wikipedia.org/wiki/Test-driven_development)
- [Performance testing best practices](https://en.wikipedia.org/wiki/Performance_testing)
