import pandas as pd

class ReadData:
    def __init__(self, extension='csv', interpreter='Python'):
        self.extension = extension
        self.interpreter = interpreter
    def read_data(self, path):
        if self.interpreter == 'PySpark' and self.extension == 'csv':
            return spark.read.option('inferSchema', 'true').option('header', 'true').csv(path)
        elif self.interpreter == 'PySpark' and self.extension == 'parquet':
            return spark.read.option('inferSchema', 'true').option('header', 'true').parquet(path)
        elif self.interpreter == 'Python' and self.extension == 'csv':
            return pd.read_csv(path)
        elif self.interpreter == 'Python' and self.extension == 'parquet':
            return pd.read_parquet(path)
        return True