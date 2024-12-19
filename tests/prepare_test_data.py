from pathlib import Path

from urfu_sber.src.common_functions import get_file_path, MODEL_TYPE
from urfu_sber.src.data_generation import main as dg
from urfu_sber.src.data_preprocessing import main as dp
from urfu_sber.src.model_preparation import main as mp

ROOT_PATH = Path.cwd()


def prepare_test_data():
    if Path(get_file_path(MODEL_TYPE.DEFAULT)).is_file() and (
            not Path(ROOT_PATH / "data" / "default").exists()
            or not Path(ROOT_PATH / "models").exists()):
        dg()
        dp()
        mp()
