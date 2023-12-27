import json
from searchtweets import load_credentials

def load_main_config():
    with open('main_configs.json', 'rb') as f:
        main_configs = json.load(f)
    return main_configs
    
def load_twitter_credentials_json():
    with open('credentials/twitter_credentials.json', 'rb') as f:
        credentials = json.load(f)
    return credentials

def load_twitter_cretentials_yaml():
    search_args = load_credentials("credentials/twitter_keys_unlimited.yaml",
                                   yaml_key="search_tweets_v2",
                                   env_overwrite=False)
    return search_args

def save_log(log_name):
    with open(f"data/logs/{log_name}.txt", "w") as f:
        f.write(f"{log_name} completed successfully!")
