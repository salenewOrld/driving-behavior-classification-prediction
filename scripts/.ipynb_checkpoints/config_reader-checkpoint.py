import yaml
class ConfigReader:
    def read(path : str):
        with open(path, 'r') as yaml_file:
            cfg : dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return cfg