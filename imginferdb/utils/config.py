import yaml


def load_config(yaml_path: str):
    with open(yaml_path, "r") as stream:
        conf = yaml.safe_load(stream)

    return conf
