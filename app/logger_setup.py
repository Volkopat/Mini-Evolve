import logging
import sys
import yaml
import os

CONFIG_FILE = "config/config.yaml"
DEFAULT_LOG_FILE = "evolution.log"
DEFAULT_LOG_LEVEL = "INFO"

# Define custom VERBOSE log level
VERBOSE_LEVEL_NUM = 15
VERBOSE_LEVEL_NAME = "VERBOSE"

# --- Cached Logger Instances ---
loggers = {}

# --- Default Configuration (used if file loading fails or section is missing) ---
DEFAULT_LOG_CONFIG = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO', # Default console level
            'stream': 'ext://sys.stdout'
        },
        # File handler can be added here if always desired, or configured via YAML
    },
    'root': { # Configuring root logger
        'handlers': ['console'],
        'level': 'INFO', # Default root level, overridden by specific loggers
    },
    'disable_existing_loggers': False
}

def load_log_config():
    """Loads logging configuration from the main YAML config file."""
    try:
        if not os.path.exists(CONFIG_FILE):
            print(f"Warning: Main config file {CONFIG_FILE} not found. Using default logging config.")
            return DEFAULT_LOG_CONFIG.get('logging', {}) # Return logging part of default

        with open(CONFIG_FILE, 'r') as f:
            full_config = yaml.safe_load(f)
        
        if 'logging' in full_config:
            return full_config['logging']
        else:
            print("Warning: 'logging' section not found in config.yaml. Using default console logging.")
            # Fallback to a simple console logger setup if 'logging' section is absent
            return { 
                "level": "INFO", 
                "file": None # Explicitly no file if not specified
            }
            
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration for logging: {e}. Using default logging config.")
        return DEFAULT_LOG_CONFIG.get('logging', {})
    except Exception as e:
        print(f"An unexpected error occurred loading logging config: {e}. Using default logging config.")
        return DEFAULT_LOG_CONFIG.get('logging', {})


def setup_logging():
    """Sets up logging based on the loaded configuration."""
    logging.addLevelName(VERBOSE_LEVEL_NUM, VERBOSE_LEVEL_NAME)

    log_config_settings = load_log_config() # This now returns the 'logging' dict from YAML or a default
    
    # Determine overall logging level
    # Ensure level_str is uppercase to match logging constants (INFO, DEBUG, etc.)
    level_str = str(log_config_settings.get('level', 'INFO')).upper()
    numeric_level = logging.getLevelName(level_str)
    if not isinstance(numeric_level, int): # Check if getLevelName returned a string (error)
        print(f"Warning: Invalid logging level '{level_str}' in config. Using INFO.")
        numeric_level = logging.INFO
        level_str = "INFO"

    # Basic formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level) # Set root logger level

    # Clear existing handlers on the root logger to avoid duplicates if this function is called multiple times
    if root_logger.hasHandlers():
        for handler in list(root_logger.handlers): # Iterate over a copy
            root_logger.removeHandler(handler)
            handler.close()


    # Console Handler (always active, level determined by config)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level) # Console handler also respects the overall level
    root_logger.addHandler(console_handler)

    # File Handler (optional, based on config)
    log_file_path = log_config_settings.get('file')
    if log_file_path:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except OSError as e:
                print(f"Error creating log directory {log_dir}: {e}. File logging disabled.")
                log_file_path = None # Disable file logging

        if log_file_path: # Re-check in case makedirs failed
            file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level) # File handler also respects the overall level
            root_logger.addHandler(file_handler)
            # print(f"Logging to file: {log_file_path} at level {level_str}") # Can be noisy

    # print(f"Console logging active at level {level_str}") # Can be noisy

# Call setup_logging() when the module is imported to configure logging immediately.
# This ensures that any logger obtained afterwards will use this configuration.
setup_logging()


def get_logger(name=None):
    """
    Retrieves a logger instance. If a name is provided, it retrieves a specific logger.
    If no name is provided, it returns the root logger.
    Ensures that loggers are configured by setup_logging() if they haven't been.
    """
    if name and name in loggers:
        return loggers[name]

    # If setup_logging hasn't been called or if root logger has no handlers,
    # it might mean initial setup didn't run or was cleared.
    # This is a basic check; robust re-configuration might be complex.
    # For this project, setup_logging() runs at import, so this is mostly a safeguard.
    # if not logging.getLogger().hasHandlers():
    # print("Re-initializing logging in get_logger (unexpected).") # Should ideally not happen
    # setup_logging() # Re-initialize if needed

    logger = logging.getLogger(name if name else "MiniEvolveApp") # Default name if None
    
    # Ensure the logger's effective level is set correctly.
    # The level set on the root logger typically propagates unless specific loggers override it.
    # No need to explicitly setLevel here if root is configured, unless overriding.

    if name:
        loggers[name] = logger
    return logger

# Example of how to use the VERBOSE level:
# logger.log(VERBOSE_LEVEL_NUM, "This is a verbose log message.")

# Example usage (and to test basic setup)
if __name__ == "__main__":
    # Test if config loading works
    print(f"Attempting to load logging config from: {CONFIG_FILE}")
    print(f"Effective Log Config: File='{DEFAULT_LOG_FILE}', Level='{DEFAULT_LOG_LEVEL}' ({getattr(logging, DEFAULT_LOG_LEVEL.upper(), logging.INFO)})")
    
    logger = get_logger("LoggerSetupTest")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    print(f"Logging test complete. Check console output and log file: '{DEFAULT_LOG_FILE}'") 