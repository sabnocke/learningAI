import pylint.config
import os

for item in pylint.config.find_default_config_files():
    print(os.path.dirname(item))
