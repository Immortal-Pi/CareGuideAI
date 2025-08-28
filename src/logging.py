import os 
import sys 
import logging 

logging_str="[%(asctime)s:%(levelname)s:%(message)s]"
log_dir='logs'
log_filepath=os.path.join(log_dir,'careguide_ai.logs')

# create handlers list 
handlers=[logging.StreamHandler(sys.stdout)]

#only add file handler if directory is writable 
try:
    os.makedirs(log_dir, exist_ok=True)
    # Test if we can weire
    with open(log_filepath,'a') as test_file:
        pass 
    handlers.append(logging.FileHandler(log_filepath))
except (OSError, PermissionError):
    # If we cant write to file, just use stdout
    pass 

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=handlers
)
logger=logging.getLogger('CAREGUIDE')