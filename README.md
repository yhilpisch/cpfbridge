# CPF Bridge — Code Resources

<p align="right">
  <img src="https://hilpisch.com/tpq_logo_bic.png" alt="The Python Quants" width="25%">
</p>

This folder collects Jupyter notebooks, Python modules, and data that
accompany four CPF bridge topics:

- covariance geometry and portfolio risk in a two-asset world;
- the Fundamental Theorems of Asset Pricing in finite-state markets;
- the Efficient Markets Hypothesis and empirical tests of return
  predictability; and
- compounding, discounting, zero-coupon bonds, and yield curves.

Each resource develops a focused, self-contained example of its topic. The
materials range from mathematical demonstrations and visualizations to
reproducible empirical diagnostics using a bundled data snapshot. They are
designed to make core definitions and relationships concrete without requiring
a full treatment of the corresponding specialization.

## Bridge Topics and Code Resources

The notebooks and modules accompany the corresponding bridge notes and slides:

- `covariance/covariance.ipynb` — covariance matrix geometry, eigenvalues,
  covariance ellipses, and portfolio variance for two assets.
- `ftap/ftap.ipynb` — two- and three-state, one-period markets, including
  replication, risk-neutral probabilities, and no-arbitrage price bounds.
- `markets/markets.ipynb` and `markets/markets.py` — the Efficient Markets
  Hypothesis, weak-form tests, and reusable return-predictability diagnostics.
- `yieldcurve/yieldcurve.ipynb` — returns, compounding, discounting,
  zero-coupon and coupon-bond valuation, and forward rates derived from an
  illustrative zero curve.
- `markets/data/eoddata.csv` and `markets/data/eoddata.meta.json` — a bundled
  end-of-day data snapshot and its source and usage metadata for offline
  diagnostics.

Open a notebook in Jupyter or Google Colab to run its examples alongside the
associated PDF note and slide deck. The yield-curve notebook uses illustrative
inputs; the market-efficiency materials use the bundled historical snapshot.

## Disclaimer

These resources are provided for educational and illustrative purposes only and come without any warranty or guarantees of any kind—express or implied. Use at your own risk. The authors and The Python Quants GmbH are not responsible for any direct or indirect damages, losses, or issues arising from the use of this code. Do not use the provided examples for critical decision‑making, financial transactions, medical advice, or production deployments without rigorous review, testing, and validation.

Some examples may reference third‑party libraries, datasets, services, or APIs subject to their own licenses and terms; you are responsible for ensuring compliance.

## Contact

- Email: team@tpq.io
- Linktree: https://linktr.ee/dyjh
- CPF Program: https://python-for-finance.com
- The AI Engineer: https://theaiengineer.dev
- The Crypto Engineer: https://thecryptoengineer.dev
- The Data Scientist: https://thedatascientist.dev
