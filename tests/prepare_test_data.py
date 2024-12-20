from pathlib import Path

from common_functions import get_file_path, MODEL_TYPE
from data_generation import main as dg
from data_preprocessing import main as dp
from model_preparation import main as mp

ROOT_PATH = Path.cwd()


def prepare_test_data():
    if Path(get_file_path(MODEL_TYPE.DEFAULT)).is_file() and (
            not Path(ROOT_PATH / "data" / "default").exists()
            or not Path(ROOT_PATH / "models").exists()):
        dg()
        dp()
        mp()
