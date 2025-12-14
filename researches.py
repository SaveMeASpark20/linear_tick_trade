"""
research.py - Model Research & Development Utilities
====================================================
Reusable boilerplate code for machine learning research and model development.
This module eliminates copy-paste by centralizing common functions for:

- Data preprocessing and time series aggregation
- Feature engineering for financial data
- Model architecture inspection and debugging
- Visualization and exploratory data analysis
- Training utilities

Import this in all your research notebooks to save time and maintain consistency.

Author: MemLabs
Course: Build a Quant Trading System
"""
import polars as pl                         # Fast dataframes for financial data
from typing import Dict, List, Tuple, Union #Type hints for function signature

#Machine learning framewrok
import torch                        #PyTorch for neural networks
import torch.nn as nn               #Neural network modules
import torch.optim as optim         #Optimization algorithms

#Numerical computing and datetime
import numpy as np                  #Numerical operations
import numpy.typing as npt               
from datetime import datetime, timedelta # Date and time handling

#Visualization
import altair                       #Interactive plotting library
import matplotlib.pyplot as plt

import random
import re
import itertools
from tqdm import tqdm
import os

SEED = 42

#=====================================================================
# TIME SERIES AGGREGATION
#=====================================================================

OHLC_AGGS = [
    #Price statistics (core OHLC)
    pl.col("price").first().alias("open"),
    pl.col("price").max().alias("high"),
    pl.col("price").min().alias("low"),
    pl.col("price").last().alias("close")
]

def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

from pathlib import Path
from typing import List, Optional

def get_trade_files(directory: str, sym: str) -> List[Path]:
    """
    Get all files in directory that stat with {sym}-trades.

    Args:
        directory: Path to directory to search
        sym: Symbol prefix (e.g., 'BTCUSDT')

    Returns:
        List of Path objects matching the pattern

    Example:
        >>> files = get_trade_files('./data', 'BTCUSDT')
        >>> # Returns: ['BTCUSDT-trades-2024.csv' 'BTCUSDT-trades-raw.parquet', ...]
    """
    dir_path = Path(directory)
    pattern = f"{sym}-trades*"
    return sorted(dir_path.glob(pattern))

def load_ohlc_timeseries(sym: str, time_interval: str):
    return load_timeseries(sym, time_interval, OHLC_AGGS)

def load_timeseries(
    sym: str,
    time_interval: str,
    aggs: List[pl.Expr],
    data_path: Optional[str] = None,
) -> pl.DataFrame: 
    """
    Load trade CSV files one by one, aggregate to time series, and concatenate.

    Args: 
        sym: symbol prefix (e.g., 'BTCUSDT')
        time_interval: Time interval for aggregration(e.g., '1h', '1m')
        aggs: List of aggregation expressions
        data_path: Optional directory path. Defaults to './cache' if not provided

    Returns:
        Concatenated time series DataFrame

    Example:
        >>> # Use default './cache' directory
        >>> ts = load_ohlc_ts('BTCUSDT', '1h', ohlc_aggs)

        >>> # Specify custom directory
        >>> ts = load_ohlc_ts('BTCUSDT', '1h', ohlc_aggs, data_path='./my_data')
    """

    #Default to './cache' if not provided
    if data_path is None:
        data_path = './cache'
    
    files = get_trade_files(data_path, sym)

    if not files:
        raise FileNotFoundError(f"No files found for {sym} in {data_path}")
    
    ts_list = []

    for file in tqdm(files, desc=f"Loading {sym}", unit="file"):
        trades = pl.read_parquet(file)

        if "datetime" not in trades.columns:
            raise ValueError(f"Column 'datetime' not found in {file.name}")
        
        trades = trades.with_columns(
            pl.col("datetime").cast(pl.Datetime)
        ).sort("datetime")

        ts = trades.group_by_dynamic(
            "datetime",
            every=time_interval,
            offset="0m"
        ).agg(aggs)

        ts_list.append(ts)
    
    result = pl.concat(ts_list)

    #Sort by datetime and remove duplicates if any
    result = result.sort("datetime").unique(subset=["datetime"])

    return result


def sharpe_annualization_factor(interval: str,
                                trading_days_per_year: int = 365,
                                trading_hours_per_day: float = 24) -> float:
    """
    Compute annualization factor (sqrt of periods per year) given a return interval.
    
    interval : str
        Frequency string like '1d', '1h', '30m', '15s'.
    trading_days_per_year : int
        Number of trading days in a year (default 252).
    trading_hours_per_day : float
        Number of trading hours in a trading day (default 6.5).
        
    Returns
    -------
    float : annualization factor
    """
    match = re.match(r"(\d+)([dhms])", interval.lower())
    if not match:
        raise ValueError("Interval must be like '1d', '2h', '15m', '30s'")
    
    value, unit = int(match.group(1)), match.group(2)
    
    # periods per year
    if unit == 'd':
        periods = trading_days_per_year / value # 365/1 = 365
    elif unit == 'h':
        periods = trading_days_per_year * (trading_hours_per_day / value) # 365 * (365 / 1h)  =  133225
    elif unit == 'm':
        periods = trading_days_per_year * (trading_hours_per_day * 60 / value) # 365 * (365 * 60 / 15m) =  532,900
    elif unit == 's':
        periods = trading_days_per_year * (trading_hours_per_day * 3600 / value) # 365 * (365 * 3600 / 30s) = 15,987,000
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    
    return np.sqrt(periods)


def plot_static_timeseries(ts: pl.DataFrame, sym: str, col: str, interval_size: str):
    plt.figure(figsize=(12,6))
    plt.plot(ts['datetime'], ts[col], label=col)
    plt.title(f'{sym} {interval_size} Bars')
    plt.xlabel('time')
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# def plot_dyn_timeseries(ts: pl.DataFrame, sym: str, col: str, time_interval: str ):
#     return altair.Chart(ts).mark_line(tooltip=True).encode(
#         x="datetime",
#         y=col
#     ).properties(
#         width=800,
#         height=400,
#         title=f"{sym} {time_interval} {col}"
#     ).configure_scale(zero=False).add_selection(
#         altair.selection_interval(bind='scales', encodings=['x']),
#         altair.selection_interval(bind='scales', encodings=['y'])
#     )


def plot_dyn_timeseries(ts: pl.DataFrame, sym: str, col: str, time_interval: str ):
    return altair.Chart(ts).mark_line(tooltip=True).encode(
        x="datetime",
        y=col
    ).properties(
        width=800,
        height=400,
        title=f"{sym} {time_interval} {col}"
    ).configure_scale(zero=False).add_selection(
        altair.selection_interval(bind='scales', encodings=['x']),  # Only zoom x-axis
        altair.selection_interval(bind='scales', encodings=['y'])   # Only zoom y-axis
    )   

def plot_column(df, col_name, figsize=(15, 6), title=None, xlabel='Index'):
    """
    Plot a column from a Polars DataFrame using matplotlib.
    
    Parameters:
    -----------
    df : polars.DataFrame
        The Polars DataFrame
    column_name : str
        Name of the column to plot
    figsize : tuple, default (15, 6)
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None, uses column name
    xlabel : str, default 'Index'
        X-axis label
    ylabel : str, optional
        Y-axis label. If None, uses column name
    """

    if title is None:
        title = col_name

    chart = df[col_name].plot.line()
    return chart.properties(
        width=800,
        height=400,
        title=title
    )
import matplotlib.pyplot as plt

def plot_column_matplotlib(df, col_name, figsize=(15, 6), title=None):
    values = df[col_name].to_numpy()

    plt.figure(figsize=figsize)
    plt.plot(values)
    plt.title(title or col_name)
    plt.xlabel("Index")
    plt.ylabel(col_name)
    plt.grid(True)
    plt.show()



import altair as alt

def plot_column_alt(df, col_name, title=None):
    """
    Plot a column from a Polars DataFrame using Altair.
    """

    # Convert to pandas because Altair works with pandas DataFrames
    pdf = df.select(col_name).to_pandas()

    # Add index column for x-axis
    pdf['index'] = range(len(pdf))

    if title is None:
        title = col_name

    chart = (
        alt.Chart(pdf)
        .mark_line()
        .encode(
            x='index:Q',
            y=f'{col_name}:Q'
        )
        .properties(
            width=800,
            height=400,
            title=title
        )
    )

    return chart


# import altair as alt

# def plot_column_alt(df, col_name, title=None):
#     """
#     Plot a column from a Polars DataFrame using Altair.
#     """
#     # Convert to pandas because altair work with pandas DataFrame

#     pdf = df.select(col_name).to_pandas()

#     pdf['index'] = range(len(pdf))

#     if title is None:
#         title = col_name

#     chart = (
#         alt.Chart(pdf)
#         .mark_line()
#         .encode(
#             x='index:Q'
#             y=f'{col_name}:Q'
#         )
#         .properties(
#             width=800,
#             height=400,
#             title=title
#         )
#     )

#     return chart


def load_ohlc_timeseries_range(
        sym: str,
        time_interval: str,
        start_date: datetime,
        end_date: datetime,
        data_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Load and aggregate trade data fro a symbol between start_date and end_date
    into OHLC time series using the give time interval

    Expects daily files named like:
        {symbol}-trades-YYYY-MM-DD.parquet
    
    Example filename:
        BTCUSDT-trades-2025-09-22.parquet

    Args:
        sym: Symbol prefix (e.g., 'BTCUSDT')
        time_interval: Aggregation interval (e.g., '1h', '5m')
        stat_date: Start datetime (inclusive)
        end_date: End datetime (inclusive)
        data_path : Directory containing cached trade parquet files (default: './cache')

    Returns:
        Poloars DataFrame with aggregated OHLC time series for the given range.
    """

    if data_path is None:
        data_path = "./cache"

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")
    
    ts_list = []
    total_days = (end_date - start_date).days + 1
    for i in tqdm(range(total_days), desc=f"Loading {sym}", unit="day"):
        current_date = start_date + timedelta(days=i)
        file_name = f"{sym}-trades-{current_date.strftime('%Y-%m-%d')}.parquet"
        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):
            tqdm.write(f"[WARNING] Missing file: {file_name}")
            continue
        
        try:
            trades = pl.read_parquet(file_path)

            if "datetime" not in trades.columns:
                raise ValueError(f"Column 'datetime' not found in {file_name}")
            
            trades = trades.with_columns(pl.col("datetime").cast(pl.Datetime))

            ts = trades.group_by_dynamic("datetime", every=time_interval, offset='0m').agg(OHLC_AGGS)
            ts_list.append(ts)

        except Exception as e:
            tqdm.write(f"[ERROR] {file_name}: {e}")
        
    if not ts_list:
        raise ValueError(f"No trade data found for {sym} in range {start_date} to {end_date}")
    
    result = pl.concat(ts_list).sort("datetime").unique(subset=["datetime"])
    return result


def add_lags(df: pl.DataFrame, col: str, max_no_lags: int, forecast_step: int) -> pl.DataFrame:
    return df.with_columns([pl.col(col).shift(i * forecast_step).alias(f"{col}_lag_{i}") for i in range(1, max_no_lags + 1)])

def plot_distribution(data: pl.DataFrame, col: str, label = None, no_bins = 100):
    return altair.Chart(data).mark_bar().encode(
        altair.X(f"{col}:Q", bin=altair.Bin(maxbins=no_bins)), y='count()'
        ).properties(
            width=600,
            height=400,
            title=f"Distribution of {label if label else col}"
        ).configure_scale(zero=False).add_params(
            altair.selection_interval(bind="scales")
        )

def print_model_info(model : torch.nn.Module, model_name: str) -> None:
    """
    Print detailed information about a PyTorch model's architecture and parameters.

    This function helps you understand:
    - Model complexity (number of parameters)
    - Which parameters are trainable vs frozen
    - Overall model architecture

    Useful for:
    - Comparing different model architectures
    - Debug training issues
    - Estimating memory requirements
    - Understanding model capacity

    Args:
        model: PyTorch model (nn.Module)
        model_name: Descriptive name for the model (e.g., 'LSTM Predictor')

    Returns:
        None (prints to console)

    Example:
        >>> model = MyTradingModel(input_size=10, hidden_size=64)
        >>> print_model_info(model, 'Trading LSTM v1')

        Output:
        Trading LSTM v1:
          Architecture: MyTradingModel(...)
          Total parameter: 15,234
          Trainable parameters: 15,234

    Note:
        Total parameters includes both trainable and frozen parameters
        For transfer learning, trainable params may be less than total.
    """

    # Count all parameters in the model
    total_params = sum(p.numel() for p in model.parameters())

    # Count only parameters that will be updated during training

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"\nArchitecture:")
    print(f"  {model}")
    print(f"\nParameter Count:")
    print(f"   Total parameters:        {total_params:,}")
    print(f"   Trainable parameters:    {trainable_params}")
    #Warn if some parameters are frozen
    if total_params != trainable_params:
        frozen_params = total_params - trainable_params
        print(f"    Frozen parameter: {frozen_params:,}")
        print(f"\n ⚠️ Note: {frozen_params:,} parameters are frozen")

    print(f"{'='*60}\n")

def print_model_params(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:\n{param.data.numpy()}")

 



def total_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def to_tensor(x, dtype=None) -> torch.Tensor:
    return torch.tensor(x.to_numpy(), dtype=torch.float32 if dtype is None else dtype)


def timeseries_split(t, test_size=0.25):
    """
    Split a tensor or array into train/test sets based on a proportion.

    Parameters
    ----------
    t: torch.Tensor or np.ndarray
        Time series data.
    test_size : float, default 0.25
        Proportion of data to use for testing. Must be between 0 and 1.
    
    Returns
    -------
    train, test : same type as t
        Train and test splits.
    
    Raises
    ------
    ValueError
        If test_size is not strictly between 0 and 1
    """

    if not (0 < test_size < 1) :
        raise ValueError(f"test_size must be between 0 and 1 (got {test_size})")
    
    split_idx = int(len(t) * (1 - test_size)) #t = (1000(X or Y len)) * (1 - 0.25) = 
    return t[:split_idx], t[split_idx:]


#X = close_log_return_lag_2
#y = close_log_return
def timeseries_train_test_split(df: pl.DataFrame, features, target, test_size=0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = df.drop_nulls()
    X = to_tensor(df[features])
    y = to_tensor(df[target]).reshape(-1, 1)
    X_train, X_test = timeseries_split(X, test_size)
    y_train, y_test = timeseries_split(y, test_size)

    return X_train, X_test, y_train, y_test    

def model_trade_results(y_true, y_pred) -> pl.DataFrame:
    """Generate trade-level results from model predictions."""

    trade_results = pl.DataFrame({
        'y_pred': y_pred.squeeze(),
        'y_true': y_true.squeeze()
    }).with_columns([
        (pl.col('y_pred').sign() == pl.col('y_true').sign()).alias('is_won'),
        pl.col('y_pred').sign().alias('position')
    ]).with_columns([
        (pl.col('position') * pl.col('y_true')).alias('trade_log_return')
    ]).with_columns([
        pl.col('trade_log_return').cum_sum().alias('equity_curve')
    ]).with_columns(
        (pl.col('equity_curve')-pl.col('equity_curve').cum_max()).alias('drawdown_log_return')
    )

    return trade_results


def eval_model_performance(y_actual, y_pred, feature_names: List[str], target_name: str, annualized_rate: float) -> Dict[str, any]:
    """Calculate performance metrics for the trading model."""

    trade_results = model_trade_results(y_actual, y_pred)

    accuracy = trade_results['is_won'].mean()
    avg_win = trade_results.filter(pl.col('is_won'))['trade_log_return'].mean()
    avg_loss = trade_results.filter(~pl.col('is_won'))['trade_log_return'].mean()
    expected_value = accuracy * avg_win + (1 - accuracy) * avg_loss
    drawdown = (trade_results['equity_curve'] - trade_results['equity_curve'].cum_max())
    max_drawdown = drawdown.min()
    sharpe = trade_results['trade_log_return'].mean() / trade_results['trade_log_return'].std() if trade_results['trade_log_return'].std() > 0  else 0
    annualized_sharpe = sharpe * annualized_rate
    equity_trough = trade_results['equity_curve'].min()
    equity_peak = trade_results['equity_curve'].max()
    total_log_return = trade_results['trade_log_return'].sum()
    std = trade_results['trade_log_return'].std()
    return {
        'features': ','.join(list(feature_names)),
        'target': target_name,
        'no_trades': len(trade_results),
        'win_rate': accuracy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'best_trade': trade_results['trade_log_return'].max(),
        'worst_trade': trade_results['trade_log_return'].min(),
        'ev': expected_value,
        'std': std,
        'total_log_return': total_log_return,
        'compound_return': np.exp(total_log_return),
        'max_drawdown': max_drawdown,
        'equity_trough': equity_trough,
        'equity_peak': equity_peak,
        'sharpe': annualized_sharpe,
    }

def batch_train_reg(
    model : nn.Module,
    X_train,
    X_test,
    y_train,
    y_test,
    no_epoch: int,
    criterion=None,
    optimizer=None,
    logging=True,
    lr=None
):
    
    # ============================================================
# 🧠 NOTES: How LBFGS Optimizer and closure() Work
# ============================================================

# LBFGS (Limited-memory BFGS) is an advanced optimization algorithm.
# It’s like gradient descent but smarter — it remembers past gradients
# to approximate the curvature (Hessian) of the loss surface.
#
# This helps it take larger, more accurate steps toward the minimum
# compared to plain gradient descent.

# ------------------------------------------------------------
# 🔹 1. closure() function
# ------------------------------------------------------------
# PyTorch’s LBFGS optimizer needs a closure — a small function that:
#    - Recomputes the loss
#    - Clears old gradients
#    - Computes new gradients (via backward pass)
# and returns the loss value.

# Example:
# def closure():
#     optimizer.zero_grad()            # 1️⃣ Clear old gradients
#     y_hat = model(X_train)           # 2️⃣ Forward pass (predictions)
#     loss = criterion(y_hat, y_train) # 3️⃣ Compute loss f(w)
#     loss.backward()                  # 4️⃣ Compute gradients ∂L/∂w, ∂L/∂b
#     return loss                      # 5️⃣ Return loss for optimizer to use

# ------------------------------------------------------------
# 🔹 2. LBFGS Optimization Step
# ------------------------------------------------------------
# When we call:
#     optimizer.step(closure)
#
# Internally, LBFGS does the following:
#
#   Step 1️⃣: Calls closure() → gets the current loss and gradient.
#   Step 2️⃣: Computes a search direction p = -H * grad
#             (H is an approximation of the inverse Hessian matrix).
#   Step 3️⃣: Performs a *line search* — it tries multiple step sizes (α)
#             along direction p to find one that satisfies the
#             Strong Wolfe conditions (not too big or small).
#             Example trial: α = 0.01, 0.05, 0.1, etc.
#   Step 4️⃣: Chooses the best α that gives the lowest loss.
#   Step 5️⃣: Updates the weights:
#                w_new = w + α * p
#   Step 6️⃣: Saves gradient & parameter deltas to improve
#             curvature estimate for future steps.

# ------------------------------------------------------------
# 🔹 3. Why Use LBFGS
# ------------------------------------------------------------
# - Converges much faster than normal SGD/Adam for small models
# - Great for regression or smaller neural nets
# - Automatically picks step size α using line search
# - Uses second-order curvature info (approximated Hessian)
#
# Drawbacks:
# - More memory use (stores past gradients)
# - Slower for large datasets or big models

# ------------------------------------------------------------
# 🔹 4. Example Concept (simple linear model)
# ------------------------------------------------------------
# Model: y = w*x + b
# X = [1,2,3], y = [2,4,6]
# Start with w=0.5, b=0.1
#
# Epoch 1:
#     closure() computes loss and grad:
#         dL/dw = -17.6
#         dL/db = -7.8
#     LBFGS tests several step sizes α:
#         α=0.01 → loss = 10.5
#         α=0.05 → loss = 7.2
#         α=0.1  → loss = 5.8 (best)
#     Update:
#         w = 0.5 + 0.1 * 17.6 = 2.26
#         b = 0.1 + 0.1 * 7.8 = 0.88
#
# Next steps:
#     LBFGS refines α and curvature info.
#     Quickly converges near w=2.0, b=0.0

# ------------------------------------------------------------
# ✅ Summary
# ------------------------------------------------------------
# LBFGS = "Smart gradient descent"
# closure() = "Recomputes loss and gradients for testing step sizes"
# Strong Wolfe = "Rule for choosing stable step length"
# Ideal for small, smooth problems like linear regression.
# ============================================================

    if criterion is None:
        criterion = nn.L1Loss()

    if lr is None:
        lr=0.0002

    #Default optimizer
    if optimizer is None:
        #Use strong_wolfe line search(more stable)
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=1,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

    if logging:
        print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters())}")
        print("Model architecture: ")
        for name, param in model.named_parameters():
            print(f"    {name}: {param.shape} ({param.numel()} params)")
        print("\nTraining model...")
    
    train_loss = None
    log_tick_size = max(no_epoch // 10, 1)

    if isinstance(optimizer, torch.optim.LBFGS):
        # LBFGS requires a closure
        for epoch in range(no_epoch):
            def closure():
                optimizer.zero_grad() # clear old gradient 
                prediction = model(X_train) 
                loss = criterion(prediction, y_train) #compute loss f(w) = (w-3)^2 
                loss.backward() # compute gradient df/dw = 2 * (w-3)
                return loss
            
            optimizer.step(closure)

            with torch.no_grad():
                train_loss = criterion(model(X_train), y_train).item()
            
            if logging and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch+1}/{no_epoch}], Loss:{train_loss:.6f}")

    else:
        # SGD/Adam loop
        for epoch in range(no_epoch):
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            if logging and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch + 1}/{no_epoch}], Loss: {loss.item():.6f}")

    if logging:
        print("\nLearned parameters: ")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}:\n {param.data.numpy()}")

    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
        print(y_hat)
        test_loss = criterion(y_hat, y_test)
        if logging:
            print(f"\nTest Loss: {test_loss.item():.6f}, Train Loss: {train_loss:.6f}")

    return y_hat



def get_linear_params(model: nn.Module) -> tuple[np.ndarray, float]:
    """Extract weights and bias from LinearModel as (w, b)."""
    weight = model.linear.weight.detach().cpu().numpy().flatten()
    bias = model.linear.bias.detach().cpu().numpy().item()
    return weight, bias

def benchmark_reg_model(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, annualized_rate, test_size=0.25, loss=None, optimizer = None, no_epochs = None, log = False, lr = None ):
    df_train, df_test = timeseries_split(df, test_size=test_size) # split data for train and test
    if no_epochs is None:
        no_epochs = 6000

    #convert the features and target to make it tensor data and make the target same shape as target
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)  
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)


    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, lr=lr, logging=log)

    perf = eval_model_performance(y_test, y_hat, features, target, annualized_rate)

    weights, biases = get_linear_params(model)
    perf['weights'] = str(weights)
    perf['bias'] = str(biases)

    return perf

# def model_trade_results(y_true, y_pred) -> pl.DataFrame:
#     """Generate trade-level results from model predictions."""
    
#     trade_results = pl.DataFrame({
#         'y_pred': y_pred.squeeze(),
#         'y_true' : y_true.squeeze()
#     }).with_columns([
#         (pl.col('y_pred').sign() == pl.col('y_true').sign()).alias('is_won'),
#         pl.col('y_pred').sign().alias('position')
#     ]).with_columns([
#         (pl.col('position') * pl.col('y_true').alias('trade_log_reuturn'))
#     ]).with_columns(
#         (pl.col('equity_curve')-pl.col("equity_curve").cum_max()).alias('drawdown_log_return')
#     )
#     return trade_results
    
def learn_model_trades(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, test_size=0.25, loss = None, optimizer = None, no_epochs = None, log = False, lr = None):
    df = df.drop_nulls()
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000

    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)

    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)

    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, criterion=loss, optimizer=optimizer, lr = lr, logging= log)

    return model_trade_results(y_test, y_hat)


def add_tx_fees_log(trades: pl.DataFrame, maker_fee, taker_fee):
    return trades.with_columns(
        (pl.col('trade_log_return') + np.log(maker_fee)).alias('trade_log_return_net_maker'),
        (pl.col('trade_log_return') + np.log(taker_fee)).alias('trade_log_return_net_taker'),
    ).with_columns(
        pl.col('trade_log_return_net_maker').cum_sum().alias('equity_curve_net_maker'),
        pl.col('trade_log_return_net_taker').cum_sum().alias('equity_curve_net_taker'),
    )


def add_tx_fees(trades: pl.DataFrame, maker_fee: float, taker_fee: float):
    trades = add_tx_fee(trades, maker_fee, 'maker')
    trades = add_tx_fee(trades, taker_fee, 'taker')
    return trades  

def add_tx_fee(trades: pl.DataFrame, tx_fee: float, name: str):
    tx_fee_col = (pl.col('exit_trade_value') * tx_fee + pl.col('entry_trade_value') * tx_fee).alias(f"tx_fee_{name}")
    return trades.with_columns(tx_fee_col)

def log_return_col(col: str) -> str:
    return f"{col}_log_return"

def log_return(col: str, shift_size: int = 1) -> pl.Expr:
    return (pl.col(col)/pl.col(col).shift(shift_size)).log().alias(log_return_col(col))

def add_log_return_features(df: pl.DataFrame, col: str, forecast_horizon: int, max_no_lags = None):
    if max_no_lags is None:
        max_no_lags = 0
    df = df.with_columns(log_return(col, forecast_horizon))
    if max_no_lags > 0:
        df = add_lags(df, log_return_col('close'), max_no_lags, forecast_horizon)

    return df

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(42)  # ensures same init every time
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def benchmark_linear_models(ts: pl.DataFrame, target: str, feature_pool: List[str], annualized_rate: int, max_no_features: int = 1, no_epochs = 200, loss = None, test_size=0.25) -> pl.DataFrame:
    import models
    
    ts = ts.drop_nulls()

    benchmarks = []
    fs = []
    for i in range(1, max_no_features+1):
        fs += list(itertools.combinations(feature_pool, i))
    
    for features in fs:
        m = models.LinearModel(len(features))
        m.apply(init_weights)
        benchmarks.append(benchmark_reg_model(ts, list(features), target, m, annualized_rate, no_epochs=no_epochs, loss=loss, test_size=test_size))

    benchmark = pl.DataFrame(benchmarks)
    return benchmark.sort('sharpe', descending=True) 

def lag_col_names(col: str, n: int) -> List[str]:
    return [f'{col}_lag_{i}' for i in range(1, n+1)]

def auto_reg_corr_matrx(df, target, max_no_lags) -> pl.DataFrame:
    return df.drop_nulls().select([target]+lag_col_names(target, max_no_lags)).corr() 

def add_model_predictions(test_trades: pl.DataFrame, model: nn.Module, features: Union[str, List[str]]) -> pl.DataFrame:
    if type(features) != list:
        features = [features]
    X_test = torch.tensor(test_trades[features].to_numpy(), dtype=torch.float32)
    y_hat = model(X_test)
    s = pl.Series('y_hat', model(X_test).detach().cpu().numpy().squeeze())
    return test_trades.with_columns(s)

# def add_equity_curve(trades: pl.DataFrame, initial_capital: float, col_name: str, suffix: str) -> pl.DataFrame:
#     return trades.with_columns(
#         (initial_capital + pl.col(col_name).cum_sum()).alias(f'equity_curve_{suffix}')
#     )

def add_equity_curve(trades: pl.DataFrame, initial_capital: float, col_name: str, suffix: str) -> pl.DataFrame:
    return trades.with_columns(
        (initial_capital + pl.col(col_name).cum_sum()).alias(f'equity_curve_{suffix}')
    )

def add_equity_curve(trades: pl.DataFrame, initial_capital: float, col_name: str, suffix: str) -> pl.DataFrame:
    return trades.with_columns(
        (initial_capital + pl.col(col_name).cum_sum()).alias(f'equity_curve_{suffix}')
    )

def add_compounding_trades(trades, capital, leverage, maker_fee, taker_fee):
    lev_capital = capital * leverage
    # calculate entry and exit trade value and size
    trades = trades.with_columns(
        ((pl.col('cum_trade_log_return').exp()) * lev_capital).shift().fill_null(lev_capital).alias('entry_trade_value'),
        ((pl.col('cum_trade_log_return').exp()) * lev_capital).alias('exit_trade_value')
    ).with_columns(
        (pl.col('entry_trade_value') / pl.col('open') * pl.col('dir_signal')).alias('signed_trade_qty'),
        (pl.col('exit_trade_value') - pl.col('entry_trade_value')).alias('trade_gross_pnl')
    )

    #add transaction fee
    trades = add_tx_fees(trades, maker_fee, taker_fee)
    
    #add net trade pnl
    trades = trades.with_columns(
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_taker')).alias('trade_net_taker_pnl'),
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_taker')).alias('trade_net_maker_pnl'),
    )
    trades = trades.with_columns(
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_taker')).alias('trade_net_taker_pnl'),
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_maker')).alias('trade_net_maker_pnl')
    )

    trades = add_equity_curve(trades, capital, 'trade_gross_pnl', 'gross')
    #add net equity curves (both taker and maker)
    trades = add_equity_curve(trades, capital, 'trade_net_taker_pnl', 'taker')
    trades = add_equity_curve(trades, capital, 'trade_net_maker_pnl', ',aker')

    return trades