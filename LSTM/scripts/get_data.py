import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        start (str): Start date for fetching data (YYYY-MM-DD).
        end (str): End date for fetching data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    df = yf.download(ticker, start=start, end=end)
    return df


def save_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to a CSV file.

    Args:
            data (pd.DataFrame): DataFrame to save.
            file_path (str): Path to save the CSV file.

    Returns:
            None
    """
    data.to_csv(file_path, index=True)
