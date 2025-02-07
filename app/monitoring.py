import logging

# Global variables to track success and failure counts
success_count = 0
failure_count = 0

# Logger setup for monitoring
logger = logging.getLogger(__name__)

def update_success_rate(success: bool):
    """
    Updates the success/failure count.
    This function does not perform relevance metric calculations.
    
    Args:
        success (bool): True if the request was successful, False otherwise.
    """
    global success_count, failure_count
    if success:
        success_count += 1
    else:
        failure_count += 1

    logger.info(f"Success count: {success_count}, Failure count: {failure_count}")

def get_monitoring_status():
    """
    Returns the success and failure counts.
    This function does not evaluate additional relevance metrics.
    
    Returns:
        dict: A dictionary containing success and failure counts.
    """
    return {
        "success_count": success_count,
        "failure_count": failure_count
    }
