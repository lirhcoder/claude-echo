"""ConfigManager - YAML-based Configuration Management System

This module provides a centralized configuration management system that:
- Loads configuration from YAML files
- Supports environment variable overrides
- Validates configuration against schemas
- Provides runtime configuration updates
- Manages configuration profiles (dev, prod, etc.)
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from pydantic import BaseModel, ValidationError
from loguru import logger
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .event_system import EventSystem, Event


class ConfigSchema(BaseModel):
    """Base configuration schema"""
    
    class Config:
        extra = "allow"  # Allow additional fields
        arbitrary_types_allowed = True


class SystemConfig(ConfigSchema):
    """System-level configuration"""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    debug: bool = False
    environment: str = "development"
    
    # Component configurations
    adapter_paths: List[str] = []
    event_system: Dict[str, Any] = {}
    security: Dict[str, Any] = {}


class AdapterConfig(ConfigSchema):
    """Adapter configuration"""
    enabled: bool = True
    priority: int = 100
    timeout: float = 30.0
    retry_count: int = 3
    config: Dict[str, Any] = {}


class SecurityConfig(ConfigSchema):
    """Security configuration"""
    risk_levels: Dict[str, List[str]] = {
        "low": ["read_file", "list_directory"],
        "medium": ["edit_file", "run_command"],
        "high": ["delete_file", "system_command"],
        "critical": ["format_drive", "network_access"]
    }
    policies: Dict[str, List[str]] = {
        "silent_mode": ["low", "medium"],
        "user_confirmation": ["high", "critical"]
    }
    audit_logging: bool = True
    encryption_key_file: Optional[str] = None


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches configuration files for changes"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml')):
            logger.info(f"Configuration file changed: {event.src_path}")
            asyncio.create_task(self.config_manager._reload_config_file(event.src_path))


class ConfigManager:
    """
    Centralized configuration management system.
    
    Features:
    - YAML configuration files
    - Environment variable overrides
    - Schema validation
    - Hot reloading of configuration files
    - Configuration profiles (dev, staging, prod)
    - Encrypted sensitive values
    """
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 environment: Optional[str] = None,
                 event_system: Optional[EventSystem] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (dev, staging, prod)
            event_system: Event system for configuration change notifications
        """
        self._config_dir = Path(config_dir or "config")
        self._environment = environment or os.getenv("ENVIRONMENT", "development")
        self._event_system = event_system
        
        # Configuration storage
        self._config: Dict[str, Any] = {}
        self._schemas: Dict[str, type] = {}
        self._watchers: Dict[str, Observer] = {}
        self._change_callbacks: Dict[str, List[Callable]] = {}
        
        # Default schemas
        self._register_default_schemas()
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the configuration manager."""
        logger.info(f"Initializing ConfigManager for environment: {self._environment}")
        
        # Create config directory if it doesn't exist
        self._config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration files
        await self._load_all_configs()
        
        # Start file watchers
        await self._start_file_watchers()
        
        self._initialized = True
        
        if self._event_system:
            await self._event_system.emit(Event(
                event_type="config_manager.initialized",
                data={"environment": self._environment, "config_dir": str(self._config_dir)}
            ))
        
        logger.info("ConfigManager initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the configuration manager."""
        logger.info("Shutting down ConfigManager")
        
        # Stop all file watchers
        for observer in self._watchers.values():
            observer.stop()
            observer.join()
        
        self._watchers.clear()
        
        if self._event_system:
            await self._event_system.emit(Event(
                event_type="config_manager.shutdown",
                data={}
            ))
        
        logger.info("ConfigManager shutdown complete")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated configuration key (e.g., 'system.log_level')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            
            # Check for environment variable override
            env_key = f"CLAUDE_ECHO_{key.replace('.', '_').upper()}"
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Try to parse the environment value
                return self._parse_env_value(env_value)
            
            return value
            
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Dot-separated configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        
        # Notify change callbacks
        self._notify_change_callbacks(key, old_value, value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        return self._config.get(section, {})
    
    def get_typed(self, key: str, schema: type, default: Any = None) -> Any:
        """
        Get a configuration value with type validation.
        
        Args:
            key: Configuration key
            schema: Pydantic model class for validation
            default: Default value if validation fails
            
        Returns:
            Validated configuration object
        """
        try:
            value = self.get(key)
            if value is None:
                return default
            
            if isinstance(value, dict):
                return schema(**value)
            else:
                return schema(value)
                
        except ValidationError as e:
            logger.error(f"Configuration validation error for key '{key}': {e}")
            return default
    
    def register_schema(self, section: str, schema: type) -> None:
        """
        Register a schema for configuration validation.
        
        Args:
            section: Configuration section name
            schema: Pydantic model class
        """
        self._schemas[section] = schema
    
    def add_change_callback(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Add a callback for configuration changes.
        
        Args:
            key: Configuration key to watch
            callback: Function called when key changes (key, old_value, new_value)
        """
        if key not in self._change_callbacks:
            self._change_callbacks[key] = []
        self._change_callbacks[key].append(callback)
    
    def remove_change_callback(self, key: str, callback: Callable) -> None:
        """
        Remove a configuration change callback.
        
        Args:
            key: Configuration key
            callback: Callback function to remove
        """
        if key in self._change_callbacks:
            self._change_callbacks[key] = [
                cb for cb in self._change_callbacks[key] if cb != callback
            ]
    
    async def reload_config(self) -> None:
        """Reload all configuration files."""
        logger.info("Reloading configuration files")
        await self._load_all_configs()
        
        if self._event_system:
            await self._event_system.emit(Event(
                event_type="config.reloaded",
                data={"environment": self._environment}
            ))
    
    async def save_config(self, section: Optional[str] = None) -> None:
        """
        Save configuration back to files.
        
        Args:
            section: Optional section to save, or None to save all
        """
        if section:
            await self._save_config_section(section)
        else:
            await self._save_all_configs()
    
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate all configuration sections against their schemas.
        
        Returns:
            Dictionary mapping section names to validation errors
        """
        errors = {}
        
        for section, schema in self._schemas.items():
            try:
                config_data = self.get_section(section)
                schema(**config_data)
            except ValidationError as e:
                errors[section] = [str(error) for error in e.errors()]
        
        return errors
    
    def _register_default_schemas(self) -> None:
        """Register default configuration schemas."""
        self.register_schema("system", SystemConfig)
        self.register_schema("security", SecurityConfig)
    
    async def _load_all_configs(self) -> None:
        """Load all configuration files."""
        # Load default configuration
        await self._load_config_file("default.yaml")
        
        # Load environment-specific configuration
        env_file = f"{self._environment}.yaml"
        await self._load_config_file(env_file)
        
        # Load local overrides (not committed to version control)
        await self._load_config_file("local.yaml")
    
    async def _load_config_file(self, filename: str) -> None:
        """
        Load a single configuration file.
        
        Args:
            filename: Name of the configuration file
        """
        config_path = self._config_dir / filename
        
        if not config_path.exists():
            logger.debug(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            
            # Merge with existing configuration
            self._merge_config(self._config, file_config)
            
            logger.debug(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file {config_path}: {e}")
    
    async def _reload_config_file(self, file_path: str) -> None:
        """
        Reload a single configuration file.
        
        Args:
            file_path: Path to the configuration file
        """
        try:
            # Reload all configs to ensure proper merging
            await self.reload_config()
            logger.info(f"Reloaded configuration after change in: {file_path}")
            
        except Exception as e:
            logger.error(f"Error reloading configuration file {file_path}: {e}")
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration dictionary to merge into
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    async def _start_file_watchers(self) -> None:
        """Start file system watchers for configuration files."""
        try:
            event_handler = ConfigFileWatcher(self)
            observer = Observer()
            observer.schedule(event_handler, str(self._config_dir), recursive=False)
            observer.start()
            self._watchers["config_dir"] = observer
            
            logger.info(f"Started configuration file watcher for: {self._config_dir}")
            
        except Exception as e:
            logger.error(f"Error starting configuration file watcher: {e}")
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """
        Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Parsed value with appropriate type
        """
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _notify_change_callbacks(self, key: str, old_value: Any, new_value: Any) -> None:
        """
        Notify all callbacks for a configuration change.
        
        Args:
            key: Configuration key that changed
            old_value: Previous value
            new_value: New value
        """
        if key in self._change_callbacks:
            for callback in self._change_callbacks[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in configuration change callback: {e}")
    
    async def _save_config_section(self, section: str) -> None:
        """
        Save a configuration section to file.
        
        Args:
            section: Section name to save
        """
        # Implementation would save to appropriate file
        # This is a simplified version
        logger.info(f"Saving configuration section: {section}")
    
    async def _save_all_configs(self) -> None:
        """Save all configuration sections."""
        # Implementation would save all sections to their respective files
        logger.info("Saving all configuration sections")


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass


class ConfigFileError(ConfigError):
    """Raised when configuration file operations fail."""
    pass