#!/usr/bin/env python3
"""
Database Configuration
Configuration settings for database operations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import os

@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    
    # Database type and connection
    db_type: str = "sqlite"  # sqlite, postgresql, mongodb
    connection_string: str = None
    
    # SQLite specific
    sqlite_path: str = "data/stock_data.db"
    
    # PostgreSQL specific
    postgresql_host: str = "localhost"
    postgresql_port: int = 5432
    postgresql_database: str = "stock_data"
    postgresql_username: str = "postgres"
    postgresql_password: str = ""
    
    # MySQL specific
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "stock_data"
    mysql_username: str = "root"
    mysql_password: str = "7874"
    
    # MongoDB specific
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_database: str = "stock_data"
    mongodb_username: str = ""
    mongodb_password: str = ""
    
    # Performance settings
    connection_pool_size: int = 10
    connection_timeout: int = 30
    query_timeout: int = 60
    
    # Data retention
    data_retention_days: int = 365  # Keep data for 1 year
    cleanup_interval_hours: int = 24  # Cleanup every 24 hours
    
    # Indexing
    enable_indexing: bool = True
    index_optimization: bool = True
    
    # Backup settings
    enable_backup: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    
    # Monitoring
    enable_monitoring: bool = True
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 1000
    
    def __post_init__(self):
        """Initialize connection string based on database type."""
        if self.connection_string is None:
            if self.db_type == "sqlite":
                self.connection_string = self.sqlite_path
            elif self.db_type == "mysql":
                self.connection_string = (
                    f"mysql://{self.mysql_username}:{self.mysql_password}"
                    f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
                )
            elif self.db_type == "postgresql":
                self.connection_string = (
                    f"postgresql://{self.postgresql_username}:{self.postgresql_password}"
                    f"@{self.postgresql_host}:{self.postgresql_port}/{self.postgresql_database}"
                )
            elif self.db_type == "mongodb":
                if self.mongodb_username and self.mongodb_password:
                    self.connection_string = (
                        f"mongodb://{self.mongodb_username}:{self.mongodb_password}"
                        f"@{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_database}"
                    )
                else:
                    self.connection_string = (
                        f"mongodb://{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_database}"
                    )
    
    def get_connection_string(self) -> str:
        """Get the connection string for the database."""
        return self.connection_string
    
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.db_type.lower() == "sqlite"
    
    def is_mysql(self) -> bool:
        """Check if using MySQL."""
        return self.db_type.lower() == "mysql"
    
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return self.db_type.lower() == "postgresql"
    
    def is_mongodb(self) -> bool:
        """Check if using MongoDB."""
        return self.db_type.lower() == "mongodb"
    
    def get_backup_path(self) -> str:
        """Get backup directory path."""
        backup_dir = Path("data/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        return str(backup_dir)
    
    def get_log_path(self) -> str:
        """Get log directory path."""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir)

# Configuration presets
CONFIG_PRESETS = {
    "development": DatabaseConfig(
        db_type="sqlite",
        sqlite_path="data/dev_stock_data.db",
        data_retention_days=30,
        cleanup_interval_hours=1,
        enable_backup=False,
        enable_monitoring=True
    ),
    
    "production": DatabaseConfig(
        db_type="mysql",
        mysql_host="localhost",
        mysql_database="stock_data_prod",
        mysql_username="root",
        mysql_password="7874",
        connection_pool_size=20,
        data_retention_days=730,  # 2 years
        cleanup_interval_hours=24,
        enable_backup=True,
        backup_interval_hours=6,
        enable_monitoring=True,
        log_slow_queries=True
    ),
    
    "cloud": DatabaseConfig(
        db_type="mongodb",
        mongodb_host="cluster.mongodb.net",
        mongodb_database="stock_data_cloud",
        connection_pool_size=50,
        data_retention_days=365,
        cleanup_interval_hours=12,
        enable_backup=True,
        backup_interval_hours=12,
        enable_monitoring=True
    ),
    
    "local": DatabaseConfig(
        db_type="mysql",
        mysql_host="localhost",
        mysql_database="stock_data",
        mysql_username="root",
        mysql_password="7874",
        data_retention_days=180,
        cleanup_interval_hours=24,
        enable_backup=True,
        backup_interval_hours=24,
        enable_monitoring=False
    )
}

def get_database_config(preset: str = "local") -> DatabaseConfig:
    """Get database configuration by preset name."""
    return CONFIG_PRESETS.get(preset, CONFIG_PRESETS["local"])

def create_custom_database_config(**kwargs) -> DatabaseConfig:
    """Create custom database configuration with overrides."""
    config = DatabaseConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

def load_config_from_env() -> DatabaseConfig:
    """Load database configuration from environment variables."""
    config = DatabaseConfig()
    
    # Override with environment variables if present
    if os.getenv("DB_TYPE"):
        config.db_type = os.getenv("DB_TYPE")
    
    if os.getenv("DB_CONNECTION_STRING"):
        config.connection_string = os.getenv("DB_CONNECTION_STRING")
    
    if os.getenv("DB_SQLITE_PATH"):
        config.sqlite_path = os.getenv("DB_SQLITE_PATH")
    
    if os.getenv("DB_POSTGRESQL_HOST"):
        config.postgresql_host = os.getenv("DB_POSTGRESQL_HOST")
    
    if os.getenv("DB_POSTGRESQL_PORT"):
        config.postgresql_port = int(os.getenv("DB_POSTGRESQL_PORT"))
    
    if os.getenv("DB_POSTGRESQL_DATABASE"):
        config.postgresql_database = os.getenv("DB_POSTGRESQL_DATABASE")
    
    if os.getenv("DB_POSTGRESQL_USERNAME"):
        config.postgresql_username = os.getenv("DB_POSTGRESQL_USERNAME")
    
    if os.getenv("DB_POSTGRESQL_PASSWORD"):
        config.postgresql_password = os.getenv("DB_POSTGRESQL_PASSWORD")
    
    if os.getenv("DB_MONGODB_HOST"):
        config.mongodb_host = os.getenv("DB_MONGODB_HOST")
    
    if os.getenv("DB_MONGODB_PORT"):
        config.mongodb_port = int(os.getenv("DB_MONGODB_PORT"))
    
    if os.getenv("DB_MONGODB_DATABASE"):
        config.mongodb_database = os.getenv("DB_MONGODB_DATABASE")
    
    if os.getenv("DB_MONGODB_USERNAME"):
        config.mongodb_username = os.getenv("DB_MONGODB_USERNAME")
    
    if os.getenv("DB_MONGODB_PASSWORD"):
        config.mongodb_password = os.getenv("DB_MONGODB_PASSWORD")
    
    # Reinitialize connection string
    config.__post_init__()
    
    return config
