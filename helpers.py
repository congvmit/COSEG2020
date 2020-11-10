from datetime import datetime
import yaml
import os

def save_args_to_file(args, save_folder):
    now = datetime.now()
    dt_string = now.strftime("config_%d-%m-%Y_%H:%M:%S.yaml")
    path_to_save = os.path.join(save_folder, dt_string)
    with open(path_to_save, 'w') as f:
        yaml.dump(vars(args), f)