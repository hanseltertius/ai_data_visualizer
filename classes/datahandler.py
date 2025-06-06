class DataHandler:
    def __init__(self, df = None):
        self.df = df

    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, value):
        self._df = value

    def remove_unnamed_columns(self, df=None):
        df = df if df is not None else self._df
        if df is not None:
            # Ensure all column names are strings to avoid errors with .str.contains
            columns_as_str = df.columns.map(str)
            mask = ~columns_as_str.str.contains('^Unnamed')
            return df.loc[:, mask]
        return None
    
    def get_clean_df(self, columns, df=None, is_get_average=False):
        df = df if df is not None else self._df
        if df is not None:
            clean_df = df.dropna(subset=columns)
            if is_get_average and len(columns) == 2:
                x_axis, y_axis = columns
                avg_df = clean_df.groupby(x_axis, dropna=False)[y_axis].mean().reset_index()
                avg_df.columns = [x_axis, y_axis]
                return avg_df
            else:
                return clean_df
        return None
    
    def get_all_columns(self, df=None):
        df = df if df is not None else self._df
        if df is not None:
            return [col for col in df.columns if not df[col].isna().all()]
        return []

    def get_numeric_columns(self, df=None):
        df = df if df is not None else self._df
        if df is not None:
            return [col for col in df.select_dtypes(include="number").columns if not df[col].isna().all()]
        return []

    def get_categorical_columns(self, df=None):
        df = df if df is not None else self._df
        if df is not None:
            return [col for col in df.select_dtypes(exclude="number").columns if not df[col].isna().all()]
        return []