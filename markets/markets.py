"""EMH testing utilities for the markets note.

This module provides reusable functions for weak-form market efficiency tests.
It is designed to work with a companion notebook (markets.ipynb) that handles
data loading, visualization, and interactive exploration.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    from statsmodels.api import OLS, add_constant
except ImportError:  # pragma: no cover
    acorr_ljungbox = None  # type: ignore[assignment]
    acf = None  # type: ignore[assignment]
    OLS = None  # type: ignore[assignment]
    add_constant = None  # type: ignore[assignment]


def to_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log-returns from a price series.

    Parameters
    ----------
    prices:
        Series of strictly positive prices indexed by date.

    Returns
    -------
    pd.Series
        Series of one-period log-returns aligned with the input index.
    """
    prices = prices.sort_index()  # ensure increasing time index
    ratio = prices / prices.shift(1)  # price relatives
    log_ret = np.log(ratio)  # one-period log-returns
    return log_ret.dropna()  # drop first NaN


def diagnostics_summary(returns: pd.Series) -> Dict[str, float]:
    """Return basic diagnostics for a log-return series.

    Diagnostics include mean, volatility, skewness, kurtosis, the number of
    observations, and the number of zero returns.
    """
    clean = returns.dropna()  # remove missing observations
    stats: Dict[str, float] = {}  # container for summary statistics
    stats["mean"] = float(clean.mean())  # average log-return
    stats["vol"] = float(clean.std())  # standard deviation of returns
    stats["skew"] = float(clean.skew())  # skewness of distribution
    stats["kurt"] = float(clean.kurt())  # kurtosis of distribution
    stats["n_obs"] = float(clean.shape[0])  # number of observations
    stats["n_zeros"] = float((clean == 0.0).sum())  # zero returns
    return stats  # return dictionary with diagnostics


def acf_ljungbox(
    returns: pd.Series,
    lags: Iterable[int] = (10, 20),
) -> pd.DataFrame:
    """Compute autocorrelation and Ljung–Box statistics for given lags.

    The output frame contains sample autocorrelations and Ljung–Box p-values
    for each requested lag.
    """
    if acorr_ljungbox is None or acf is None:  # pragma: no cover
        msg = "statsmodels is required for acf_ljungbox"
        raise ImportError(msg)

    clean = returns.dropna()  # remove missing values
    max_lag = max(lags)  # largest lag for autocorrelation
    acf_vals = acf(
        clean,
        nlags=max_lag,
        fft=True,
        missing="drop",
    )  # autocorrelation estimates

    lb = acorr_ljungbox(
        clean,
        lags=list(lags),
        return_df=True,
    )  # Ljung–Box statistics

    # build a compact summary frame
    frame = pd.DataFrame(
        {
            "lag": list(lags),
            "acf": [acf_vals[k] for k in lags],
            "lb_stat": lb["lb_stat"].to_numpy(),
            "lb_pvalue": lb["lb_pvalue"].to_numpy(),
        }
    ).set_index("lag")
    return frame


def runs_test(returns: pd.Series) -> Dict[str, float]:
    """Perform a simple runs test on return signs.

    A low p-value suggests too few or too many sign changes relative to a
    benchmark of independent signs. This function uses a basic normal
    approximation to keep dependencies minimal.
    """
    clean = returns.dropna()  # remove missing values
    signs = np.sign(clean.to_numpy())  # +1, 0, -1 signs
    signs = signs[signs != 0.0]  # drop exact zeros

    if signs.size == 0:
        return {"z_stat": np.nan, "p_value": np.nan}

    # count runs of consecutive identical signs
    runs = 1  # first observation starts the first run
    for i in range(1, signs.size):
        if signs[i] != signs[i - 1]:
            runs += 1  # new run starts

    n_pos = int((signs > 0.0).sum())  # number of positive returns
    n_neg = int((signs < 0.0).sum())  # number of negative returns

    if n_pos == 0 or n_neg == 0:
        return {"z_stat": np.nan, "p_value": np.nan}

    # expected runs and variance under randomness
    n = n_pos + n_neg  # total number of observations
    mu_r = 1.0 + 2.0 * n_pos * n_neg / n  # expected number of runs
    var_r = (
        2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)
        / (n * n * (n - 1.0))
    )  # variance of runs

    z_stat = (runs - mu_r) / np.sqrt(var_r)  # standardized test statistic
    # two-sided p-value from normal approximation
    from math import erf, sqrt  # import locally to avoid global state

    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z_stat) / sqrt(2.0))))
    return {"z_stat": float(z_stat), "p_value": float(p_value)}


def variance_ratio(
    returns: pd.Series,
    q_list: Iterable[int] = (2, 5, 10, 20),
) -> pd.DataFrame:
    """Compute simple variance ratio statistics for several horizons.

    Parameters
    ----------
    returns:
        Series of one-period log-returns.
    q_list:
        Iterable of integer horizons q for which to compute VR(q).

    Returns
    -------
    pd.DataFrame
        Frame indexed by horizon with columns ``vr`` and ``n_eff`` giving the
        variance ratio and the effective number of non-overlapping blocks.
    """
    clean = returns.dropna()  # remove missing values
    var1 = float(clean.var(ddof=1))  # sample variance of one-period returns

    records: List[Tuple[int, float, int]] = []  # container for results

    for q in q_list:
        q_int = int(q)  # ensure integer horizon
        if q_int <= 1 or q_int > clean.shape[0]:
            continue  # skip invalid horizons

        # build non-overlapping q-period sums
        n_block = clean.shape[0] // q_int  # number of full blocks
        reshaped = clean.iloc[: n_block * q_int].to_numpy().reshape(
            n_block,
            q_int,
        )  # blocks of length q_int
        summed = reshaped.sum(axis=1)  # q-period returns
        var_q = float(np.var(summed, ddof=1))  # sample variance of q-period sum
        vr = var_q / (q_int * var1)  # variance ratio
        records.append((q_int, vr, n_block))  # store result

    frame = pd.DataFrame(
        records,
        columns=["q", "vr", "n_eff"],
    ).set_index("q")
    return frame


def predictability_regression(
    returns: pd.Series,
    p: int=1,
    hac_lags: int=5,
) -> pd.Series:
    """Estimate an autoregression of returns with HAC-robust inference.

    Parameters
    ----------
    returns:
        Series of one-period log-returns.
    p:
        Lag order for the autoregression.
    hac_lags:
        Maximum lag for the HAC covariance estimator.

    Returns
    -------
    pd.Series
        Series with coefficient estimates, t-statistics, and p-values.
    """
    if OLS is None or add_constant is None:  # pragma: no cover
        msg = "statsmodels is required for predictability_regression"
        raise ImportError(msg)

    clean = returns.dropna()  # remove missing values
    df = pd.DataFrame({"r": clean})  # container for lags
    for k in range(1, p + 1):
        df[f"lag_{k}"] = df["r"].shift(k)  # construct lagged returns
    df = df.dropna()  # drop rows with incomplete lags

    y = df["r"]  # dependent variable
    x = df[[f"lag_{k}" for k in range(1, p + 1)]]  # regressors
    x = add_constant(x)  # add intercept

    model = OLS(y, x)  # ordinary least squares model
    results = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )  # HAC-robust fit

    out = pd.Series(dtype=float)  # container for output
    for name, coef in results.params.items():
        out[f"coef_{name}"] = float(coef)  # coefficient estimate
    for name, tval in results.tvalues.items():
        out[f"t_{name}"] = float(tval)  # t-statistic
    for name, pval in results.pvalues.items():
        out[f"p_{name}"] = float(pval)  # p-value
    out["r2"] = float(results.rsquared)  # R-squared of regression
    return out


def oos_forecast_eval(
    returns: pd.Series,
    window: int=252,
    costs_bps: float=2.0,
) -> Dict[str, float]:
    """Evaluate a simple rolling AR(1) forecast and toy trading strategy.

    The strategy uses a rolling window to estimate an AR(1) model and then
    takes positions based on the sign of the one-step-ahead forecast. Net
    performance accounts for a symmetric round-trip cost specified in basis
    points of notional per trade.
    """
    clean = returns.dropna()  # remove missing values
    if clean.shape[0] <= window + 1:
        return {
            "mse": np.nan,
            "hit_rate": np.nan,
            "gross_pnl": np.nan,
            "net_pnl": np.nan,
        }

    forecasts: List[float] = []  # container for forecasts
    realized: List[float] = []  # container for realized returns
    positions: List[float] = []  # container for trading positions

    for end in range(window, clean.shape[0] - 1):
        sample = clean.iloc[end - window : end]  # estimation window
        # simple AR(1) coefficient via least-squares
        x = sample.shift(1).dropna().to_numpy()  # lagged returns
        y = sample.loc[sample.index[1:]].to_numpy()  # aligned returns
        if x.size == 0:
            continue  # skip if not enough data
        beta = float(np.dot(x, y) / np.dot(x, x))  # AR(1) slope
        r_t = float(clean.iloc[end])  # last observed return
        forecast = beta * r_t  # one-step-ahead forecast

        r_next = float(clean.iloc[end + 1])  # realized next-period return
        pos = float(np.sign(forecast))  # position based on forecast sign

        forecasts.append(forecast)
        realized.append(r_next)
        positions.append(pos)

    if not forecasts:
        return {
            "mse": np.nan,
            "hit_rate": np.nan,
            "gross_pnl": np.nan,
            "net_pnl": np.nan,
        }

    f_arr = np.asarray(forecasts)  # array of forecasts
    r_arr = np.asarray(realized)  # array of realized returns
    p_arr = np.asarray(positions)  # array of positions

    mse = float(np.mean((f_arr - r_arr) ** 2))  # mean-squared error
    hit_rate = float(
        np.mean(np.sign(f_arr) == np.sign(r_arr))
    )  # frequency of correct sign

    gross_ret = p_arr * r_arr  # gross strategy returns
    turn = np.abs(np.diff(p_arr, prepend=0.0))  # position changes
    cost_per_trade = costs_bps * 1e-4  # round-trip cost in return space
    costs = cost_per_trade * turn  # trading costs per period
    net_ret = gross_ret - costs  # net strategy returns

    gross_pnl = float(gross_ret.sum())  # total gross PnL
    net_pnl = float(net_ret.sum())  # total net PnL

    return {
        "mse": mse,
        "hit_rate": hit_rate,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
    }


def efficiency_scorecard(
    returns_dict: Dict[str, pd.Series],
    lb_lags: Iterable[int] = (10, 20),
    vr_q: Iterable[int] = (2, 5, 10, 20),
) -> pd.DataFrame:
    """Build a compact efficiency scorecard for several assets.

    For each asset, the scorecard includes basic distributional diagnostics,
    short-lag autocorrelation and Ljung–Box p-values, variance ratios, and a
    simple out-of-sample AR(1) evaluation.
    """
    rows: List[pd.Series] = []  # container for per-asset summaries

    for name, rets in returns_dict.items():
        diag = diagnostics_summary(rets)  # distributional diagnostics
        row = pd.Series(diag)  # start with diagnostics

        try:
            acf_lb = acf_ljungbox(rets, lags=lb_lags)  # autocorrelation stats
            for lag in lb_lags:
                row[f"acf_{lag}"] = float(acf_lb.loc[lag, "acf"])
                row[f"lb_p_{lag}"] = float(acf_lb.loc[lag, "lb_pvalue"])
        except Exception:  # pragma: no cover
            pass  # keep diagnostics even if autocorrelation fails

        try:
            vr = variance_ratio(rets, q_list=vr_q)  # variance ratios
            for q in vr.index:
                row[f"vr_{q}"] = float(vr.loc[q, "vr"])
        except Exception:  # pragma: no cover
            pass  # proceed even if variance ratio computation fails

        oos = oos_forecast_eval(rets)  # out-of-sample AR(1) evaluation
        for key, value in oos.items():
            row[f"oos_{key}"] = float(value)

        row.name = name  # label row by asset name
        rows.append(row)

    if not rows:
        return pd.DataFrame()  # empty frame if no assets

    scorecard = pd.DataFrame(rows)  # assemble frame from rows
    return scorecard


def plot_suite(
    prices: pd.Series,
    returns: pd.Series,
    title: Optional[str]=None,
) -> None:
    """Plot a basic diagnostic suite for prices and returns.

    The plot includes price and log-price panels, the return series, and a
    rolling volatility estimate based on a fixed window.
    """
    log_price = np.log(prices)  # log-prices from prices
    roll_vol = returns.rolling(window=50).std()  # rolling volatility

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
    ax_price = axes[0, 0]
    ax_log = axes[0, 1]
    ax_ret = axes[1, 0]
    ax_vol = axes[1, 1]

    ax_price.plot(prices.index, prices.values, color="tab:blue")
    ax_price.set_title("Price")

    ax_log.plot(log_price.index, log_price.values, color="tab:orange")
    ax_log.set_title("Log-Price")

    ax_ret.plot(returns.index, returns.values, color="tab:green")
    ax_ret.set_title("Log-Returns")

    ax_vol.plot(roll_vol.index, roll_vol.values, color="tab:red")
    ax_vol.set_title("Rolling Volatility")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()  # adjust layout for readability
    plt.show()

