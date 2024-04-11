import sys
from pathlib import Path

project_folder = Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder))
from process_experiments_pg import nyc_experiment
from sklearn.linear_model import LinearRegression


def linearregression_complex():

    df = nyc_experiment("deep", LinearRegression(), [1, 10000, 100000, 729000])

    return df
