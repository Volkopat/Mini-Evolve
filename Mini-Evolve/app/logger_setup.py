import logging
import sys
import yaml

CONFIG_FILE = "config/config.yaml"
DEFAULT_LOG_FILE = "evolution.log"
DEFAULT_LOG_LEVEL = "INFO"

def load_log_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('logging', {})
    except FileNotFoundError:
        # print(f"Warning: Main configuration file {CONFIG_FILE} not found. Using default logging settings.")
        return {}
    except yaml.YAMLError as e:
        # print(f"Warning: Error parsing YAML for logging config: {e}. Using default settings.")
        return {}

log_config = load_log_config()
log_file = log_config.get('file', DEFAULT_LOG_FILE)
log_level_str = log_config.get('level', DEFAULT_LOG_LEVEL).upper()

log_level = getattr(logging, log_level_str, logging.INFO)

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a'), # Append mode
        logging.StreamHandler(sys.stdout) # Log to console
    ]
)

def get_logger(name: str):
    """Returns a logger instance with the pre-configured settings."""
    return logging.getLogger(name)

# Example usage (and to test basic setup)
if __name__ == "__main__":
    # Test if config loading works
    print(f"Attempting to load logging config from: {CONFIG_FILE}")
    print(f"Effective Log Config: File='{log_file}', Level='{log_level_str}' ({log_level}))")
    
    logger = get_logger("LoggerSetupTest")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    print(f"Logging test complete. Check console output and log file: '{log_file}'") 