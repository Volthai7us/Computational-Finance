import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


def pearson_correlation(df: pd.DataFrame) -> pd.Series:
    """
    Add Pearson correlation to DataFrame.
    Pearson correlation measures the linear correlation between two variables.
    """
    numeric_columns = df.select_dtypes(include=['number'])
    correlations = numeric_columns.corrwith(df['close_price'])

    return correlations


def random_forest_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Random Forest feature importance to DataFrame.
    Random Forest feature importance measures the importance of each feature
    when predicting the target variable.
    """

    numeric_columns = df.select_dtypes(include=['number'])
    X = numeric_columns.drop('close_price', axis=1)
    y = numeric_columns['close_price']

    model = RandomForestRegressor()
    model.fit(X, y)

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    return importance_df.sort_values('Importance', ascending=False)


def rfe_feature_importance(df: pd.DataFrame, n_features_to_select: int = 5) -> pd.Index:
    """
    Add Recursive Feature Elimination feature importance to DataFrame.
    Recursive Feature Elimination feature importance measures the importance
    of each feature when predicting the target variable.
    """

    numeric_columns = df.select_dtypes(include=['number'])
    X = numeric_columns.drop('close_price', axis=1)
    y = numeric_columns['close_price']

    model = LinearRegression()
    selector = RFE(model, n_features_to_select=n_features_to_select)
    selector = selector.fit(X, y)

    selected_features = X.columns[selector.support_]

    return selected_features
