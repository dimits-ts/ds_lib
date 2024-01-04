from sklearn.model_selection import train_test_split
import pandas as pd


def train_test_val_split(df: pd.DataFrame, 
                         train_ratio: float=0.8, 
                         val_ratio: float=0.1, 
                         test_ratio: float=0.1,
                         random_state: int=None, 
                         stratify_col: str=None
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training, validation, and test sets.
    
    :param df: The input DataFrame to be split.
    :type df: pd.DataFrame
    
    :param train_ratio: The proportion of the data to include in the training set (default: 0.8).
    :type train_ratio: float
    
    :param val_ratio: The proportion of the data to include in the validation set (default: 0.1).
    :type val_ratio: float
    
    :param test_ratio: The proportion of the data to include in the test set (default: 0.1).
    :type test_ratio: float
    
    :param random_state: Seed for the random number generator to ensure reproducibility (default: None).
    :type random_state: int
    
    :param stratify_col: Column name for stratified sampling. If provided, it ensures that
        the proportions of the target variable are the same in each split (default: None).
    :type stratify_col: str
    
    :return: A tuple containing three DataFrames - (df_train, df_val, df_test).
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    
    :raises AssertionError: Raised if the sum of train_ratio, val_ratio, and test_ratio is not equal to 1.
    
    Examples:
    ```python
    # Without stratified sampling
    train, val, test = train_test_val_split(my_dataframe, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    
    # With stratified sampling
    train, val, test = train_test_val_split(my_dataframe, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, stratify_col='target_column')
    ```
    """
    assert sum([train_ratio, val_ratio, test_ratio]) == 1, "Split ratios must sum to 1"
    
    stratify = df[stratify_col] if stratify_col is None else None
    df_train, df_temp = train_test_split(df, 
                                        test_size=1-train_ratio, 
                                        random_state=random_state, 
                                        stratify=stratify)
    
    stratify = df_temp[stratify_col] if stratify_col is None else None
    df_val, df_test = train_test_split(df_temp,
                                       test_size=test_ratio/(test_ratio + val_ratio),
                                       random_state=random_state,
                                       stratify=stratify) 
    return df_train, df_val, df_test