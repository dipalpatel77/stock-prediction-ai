# Services — Quick Reference

All services live in `main/services/`. Import via `from main.services import X` or directly from the module.

## Data & API Services

| File | Class | Role | Key Consumer |
|------|-------|------|-------------|
| `data_service.py` | `DataService` (via `DataServiceWrapper`) | Unified entry point for stock data loading | `data_processor.py` |
| `angel_one_service.py` | `AngelOneService` | Angel One SmartAPI client (NSE/Indian stocks) | `angel_one_manager.py` |
| `angel_one_manager.py` | `AngelOneManager` | Session/token lifecycle wrapper for Angel One | `api_coordinator.py` |
| `api_coordinator.py` | `APICoordinator` | Multi-source API orchestration with circuit breaking, rate limiting, caching, and fallbacks | `data_processor.py` |
| `multi_exchange_data_service.py` | `MultiExchangeDataService` | Aggregates data from multiple exchanges | `api_coordinator.py` |
| `fred_api_service.py` | `FREDAPIService` | FRED economic indicator API client | `api_coordinator.py` |
| `global_market_service.py` | `GlobalMarketService` | Global market indices (Dow, NASDAQ, FTSE, Nikkei) | `api_coordinator.py` |
| `currency_service.py` | `CurrencyService` | Real-time FX rates, 30+ currency pairs | `report_generator.py` |

## Incremental Update System (three cooperating services)

| File | Class | Role |
|------|-------|------|
| `incremental_data_service.py` | `IncrementalDataService` | **Fetches** raw OHLCV data from external APIs; handles gap detection |
| `incremental_service.py` | `IncrementalService` | **Retrains** ML models incrementally with new data |
| `incremental_update_service.py` | `IncrementalUpdateService` | **Decides** whether to do full refresh vs incremental append; used directly by `data_processor.py` |

## ML & Model Services

| File | Class | Role | Key Consumer |
|------|-------|------|-------------|
| `model_service.py` | `ModelService` | Model training, prediction, ensemble management | `model_trainer.py` |
| `feature_engineering_service.py` | `FeatureEngineeringService` | Feature extraction and transformation | `data_processor.py` |
| `technical_indicators_service.py` | `TechnicalIndicatorsService` | RSI, MACD, Bollinger Bands, ADX, etc. | `data_processor.py` |
| `validation_predictor.py` | `ValidationPredictor` | Prediction accuracy validation and backtesting | `core_pipeline.py` |

## Database

| File | Class | Role |
|------|-------|------|
| `database_manager.py` | `DatabaseManager` | Singleton DB manager; connection pooling, query optimization, async support. Default DB: `main/data/default.db` |
| `news_sentiment_database_manager.py` | `NewsSentimentDatabaseManager` | Persists news sentiment scores |

## News & Sentiment

| File | Class | Role |
|------|-------|------|
| `simple_news_sentiment_service.py` | `SimpleNewsSentimentService` | **Active** news sentiment implementation (VADER + TextBlob) |
| `rss_news_service.py` | `RSSNewsService` | Collects headlines from RSS feeds |
| `news_sentiment_service.py` | *(deleted — was an empty stub)* | — |

## Utilities & Infrastructure

| File | Class | Role |
|------|-------|------|
| `interval_manager.py` | `IntervalManager` | Maps prediction horizons to data intervals (ONE_MINUTE, ONE_DAY, etc.) |
| `report_generator.py` | `ReportGenerator` | Generates JSON/HTML/CSV/TXT reports with multi-currency formatting |
| `data_service_wrapper.py` | `DataServiceWrapper` | Thin compatibility wrapper around the main data service |
