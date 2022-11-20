import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

class AssignTransformer(BaseEstimator, TransformerMixin):
    """
    Class for applying `DataFrame.assign`, using a dict as
    unique parameter
    """
    def __init__(self, assign_map, copy=True):
        self.assign_map = assign_map
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy() if self.copy else X
        return X_.assign(**self.assign_map)

class NameTransformer(BaseEstimator, TransformerMixin):
    """
    Class used to select and rename the columns of a DataFrame.
    """

    def __init__(self, names_map, keep_features, copy=True):
        if isinstance(keep_features, list):
            intersection_ = set(names_map.keys()).intersection(keep_features)
            assert len(intersection_) == 0, \
                f"""The following columns are defined both in `keep_features`
                and in `names_map.keys()`:\n{intersection_}
                """
        self.names_map = names_map
        self.keep_features = keep_features
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def copy_if(df, condition):
            return df.copy() if condition else df

        # if `keep_features` is list, keep only renamed and listed columns
        # if `keep_features` is False, keep only renamed variables
        if isinstance(self.keep_features, list) or not self.keep_features:
            columns = list(self.names_map.keys()) + (self.keep_features or [])
            X_ = copy_if(X[columns], self.copy)
        else:
            # if `keep_features` is True, keep all variables
            X_ = copy_if(X, self.copy)
        X_ = X_.rename(self.names_map, axis=1)
        return X_

class MergeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to merge two pandas Data Frames
    """

    def __init__(
            self, right_df, keep_index=True, merge_kwargs=None, copy=True):
        if merge_kwargs is None:
            merge_kwargs = {}
        self.right_df = right_df
        self.keep_index = keep_index
        self.merge_kwargs = merge_kwargs.copy()
        self.copy = copy

    def fit(self, df):
        return self

    def transform(self, df):
        df_ = df.copy() if self.copy else df
        if self.keep_index:
            index_names = df_.index.names
            # handle unnamed index
            is_unnamed_index = (
                    (index_names[0] is None) and
                    (len(index_names) == 1)
            )
            index_names = ['index'] if is_unnamed_index else index_names
            # merge respecting index and `merge_kwargs`
            df_ = df_.reset_index() \
                     .merge(self.right_df, **self.merge_kwargs) \
                     .set_index(index_names)
            if is_unnamed_index:
                df_.index.names = [None]
        else:
            df_ = df_.merge(self.right_df, **self.merge_kwargs)
        return df_


class OneHotPandas(OneHotEncoder):
    def __init__(self, categories='auto', drop=None, sparse=False,
                 dtype=np.int, handle_unknown='ignore', key_var=None ):
        # OneHotEncoder originals
        super().__init__(categories=categories, drop=drop, sparse=sparse,
                         dtype=dtype, handle_unknown=handle_unknown)

        # In order to keep pandas column names
        self.original_names = None
        self.feature_names = None
        self.key_var = key_var

    def clean_feature_names(self, cols):
        new_names = []
        for col in cols:
            new_names.append(col.replace(".0", ""))
        return new_names

    def fit(self, X, y=None):
        _X = X.set_index(self.key_var)
        _fit = super().fit(_X)
        self.original_names = self.clean_feature_names(
            _fit.get_feature_names(list(_X.columns)))
        return _fit

    def transform(self, X, y=None):
        _transform = pd.DataFrame(
            super().transform(X.set_index(self.key_var)),
            dtype=self.dtype,
            columns=self.original_names)
        self.feature_names= [f for f in self.original_names
                               if not ('9999' in f or '99' in f)]

        return pd.concat(
            [X[[self.key_var]], _transform[self.feature_names]],
            axis=1)
