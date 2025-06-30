import logging
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, 'Log')

os.makedirs(log_dir, exist_ok=True)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s')

app_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'), mode='w')
app_handler.setLevel(logging.INFO)
app_handler.setFormatter(formatter)

app_logger = logging.getLogger('app_logger')
app_logger.setLevel(logging.INFO)
app_logger.addHandler(app_handler)
