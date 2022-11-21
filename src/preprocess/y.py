import numpy as np


INDEX_KEY = 'idhogar'

NAMES_MAP = {
    'target': 'y'
}

KEEP_FEATURES = [INDEX_KEY]

ASSIGN_MAP = {
    # binarize target variable
    'y': lambda df: np.where(
        df['y'].isin([1, 2]),
        1,  # general poverty
        0   # no poverty
    )
}
