import numpy as np
import pandas as pd
from urfu_sber.src.common_functions import calculate_scor, normalize_df


def test_calculate_score_direct_dependency():
    df_data = pd.DataFrame({
        'criterion1': [10],
        'criterion2': [20]
    })

    df_params = pd.DataFrame({
        'name': ['criterion1', 'criterion2'],
        'weight': [2, 3],
        'direct_dependence': [1, 1],
        'max_value': [100, 100]
    })

    expected_score = (10 * 0.4) + (20 * 0.6)

    actual_score = calculate_scor(df_data, df_params)

    assert abs(actual_score - expected_score) < 0.0001


def test_normalize_weights_division():
    input_df = pd.DataFrame({
        'name': ['criterion1', 'criterion2', 'criterion3'],
        'weight': [2, 3, 5]
    })
    expected_weights = np.array([0.2, 0.3, 0.5])

    result_df = normalize_df(input_df)

    np.testing.assert_array_almost_equal(
        result_df['weight'].values,
        expected_weights,
        decimal=2
    )
