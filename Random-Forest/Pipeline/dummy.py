import logging

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the script execution.")
    
    # Your script logic here
    try:
        result = 10 / 2  # Example computation
        logger.info(f"Computation result: {result}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    logger.info("Script execution completed.")

if __name__ == "__main__":
    main()
