import os
import string
import hashlib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
    
    
    
def generate_rule_id(value):
    """
    Generate a deterministic 3-character alphanumeric code from a given string.

    The function uses a SHA-256 hash of the input string to produce a
    reproducible code composed of uppercase letters (A–Z) and digits (0–9).
    For the same input value, the output will always be identical, independent
    of platform or Python version.

    Parameters
    ----------
    value : str
        Input string from which the code is deterministically derived.

    Returns
    -------
    str
        A 3-character alphanumeric code (uppercase letters and digits).

    Notes
    -----
    This method does not rely on Python's random number generator and is
    therefore suitable for stable identifiers, testing, and reproducible
    workflows. Collisions are possible due to the short code length but are
    unlikely for small input sets.
    """
    chars = string.ascii_uppercase + string.digits
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()

    # Convert hash to an integer and map to the character set
    num = int(digest, 16)
    code = ""
    for _ in range(3):
        num, idx = divmod(num, len(chars))
        code += chars[idx]

    return code


# utility class & functions
class ProphetUtility:
    @staticmethod
    def stan_init(model):
        """
        Retrieve parameters from a trained model having format used to initialize a new Stan model.
        :param model: a trained model of the Prophet class.
        :return: a dictionary containing retrieved parameters of m.
        """
        res = {}
        for pname in ['k', 'm', 'sigma_obs']:
            res[pname] = model.params[pname][0][0]
        for pname in ['delta', 'beta']:
            res[pname] = model.params[pname][0]
        return res

    @staticmethod
    class suppress_stdout_stderr(object):
        """
        A context manager for doing a "deep suppression" of stdout and stderr in Python, i.e. will suppress all print,
        even if the print originates in a compiled C/Fortran sub-function.
        This will not suppress raised exceptions, since exceptions are printed to stderr just before a script exits,
        and after the context manager has exited (at least, I think that is why it lets exceptions through).
        https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
        """

        def __init__(self):
            # Open a pair of null files
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
            # Save the actual stdout (1) and stderr (2) file descriptors.
            self.save_fds = (os.dup(1), os.dup(2))

        def __enter__(self):
            # Assign the null pointers to stdout and stderr.
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

        def __exit__(self, *_):
            # Re-assign the real stdout/stderr back to (1) and (2)
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            # Close the null files
            os.close(self.null_fds[0])
            os.close(self.null_fds[1])


def orthogonality_check(dml_estimator, random_state=None):
    """
    Empirical orthogonality check for a fitted Double Machine Learning (DML) estimator.

    The test evaluates whether the cross-fitted residuals produced by the DML
    procedure are approximately orthogonal to the residualized covariates and
    controls. Orthogonality is assessed by testing whether these residuals can
    be predicted from the residualized feature matrix using a flexible
    non-linear model.

    Specifically, a Random Forest regressor is trained in a cross-validation
    setting to predict:
        (i) outcome residuals (y_res)
        (ii) treatment residuals (T_res)

    from the stacked residualized covariates and controls. A non-negligible
    out-of-sample R² indicates residual dependence and therefore a potential
    violation of the orthogonality condition.

    Parameters
    ----------
    dml_estimator : object
        A fitted DML estimator exposing the attribute `residuals_`,
        expected to contain (y_res, T_res, X, W) as cross-fitted residuals
        and residualized covariates/controls.
    random_state : int or None, optional
        Random seed used for cross-validation splits and the Random Forest
        estimator, by default None.

    Returns
    -------
    result : bool
        True if the orthogonality condition is deemed satisfied
        (i.e. cross-validated R² ≤ 0.01 for both outcome and treatment residuals),
        False otherwise.
    """

    # Extract cross fitted residuals from the DML estimator
    y_res , T_res, X, W = dml_estimator.residuals_   

    # Build regressor matrix using residualized covariates and controls
    Z = X if W is None else np.hstack([X, W])
    
    result = True
    for res in [y_res, T_res]:
        
        # Ensure treatment residuals are two dimensional
        res = np.asarray(res)
        res = res.reshape(-1, 1) if res.ndim == 1 else res
            
        # Cross validation configuration
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state) 

        # Non linear critic model used to detect residual dependence
        min_samples_leaf = int(np.ceil(0.01 * len(res)))    # Base heuristic 
        rf = RandomForestRegressor(n_estimators=100,
                                   max_depth=10,
                                   min_samples_leaf=min_samples_leaf,
                                   min_samples_split=int(2 * min_samples_leaf),
                                   bootstrap=True,
                                   max_samples=0.8,
                                   n_jobs=-1,
                                   random_state=random_state)
    
        # Orthogonality test
        r2 = cross_val_score(rf, Z, res, cv=cv, scoring="r2").mean()

        # R2 above threshold indicates violation of orthogonality
        if r2 > 0.01:
            result = False
            break
            
    return result


def positivity_check(T, X, alpha=0.05, random_state=None):
    """
    Check the positivity (overlap) assumption for a binary treatment.

    The function estimates propensity scores via logistic regression and
    verifies that treated and control units have sufficient overlap.
    Positivity is considered violated if a large fraction of units have
    extreme propensity scores or if the common support between groups
    is too narrow.

    Parameters
    ----------
    T : array-like
        Binary treatment indicator (0/1).
    X : array-like
        Covariates used to estimate the propensity score.
    alpha : float, default=0.05
        Threshold defining extreme propensity scores.

    Returns
    -------
    result : bool
        True if the positivity assumption is satisfied, False otherwise.
    scores : ndarray
        Estimated propensity scores for each observation.
    """

    # Ensure treatment is a 1D binary array
    T = np.asarray(T).ravel().astype(int)

    # Estimate propensity scores using regularized logistic regression
    propensity_model = LogisticRegression(penalty="l2",
                                          C=1.0,
                                          solver="lbfgs",
                                          max_iter=2000,
                                          n_jobs=-1,
                                          random_state=random_state)

    scores = propensity_model.fit(X, T).predict_proba(X)[:, 1]

    # Fraction of observations with extreme propensity scores
    frac_low = np.mean(scores < alpha)
    frac_high = np.mean(scores > 1 - alpha)

    # Propensity score distributions for treated and control units
    scores_treated = scores[T == 1]
    scores_control = scores[T == 0]

    # Common support interval between treated and control groups
    support_lo = max(scores_treated.min(), scores_control.min())
    support_hi = min(scores_treated.max(), scores_control.max())
    support_width = support_hi - support_lo

    # Positivity rules:
    # 1) No more than 10% of observations with extreme propensity scores
    # 2) Non-trivial common support between treated and controls    
    
    result = True
    
    if frac_low > 0.10 or frac_high > 0.10 or support_width <= 0.20:
        result = False

    return result, scores


def display_processed_data(processed_data, n=3):
    
    head, tail = processed_data.head(n), processed_data.tail(n)
    sep = pd.DataFrame([["…"] * processed_data.shape[1]], columns=processed_data.columns, index=["…"])
    view = pd.concat([head, sep, tail])

    return (view.style
                 .format({
                     'date_stamp': lambda x: "…" if (pd.isna(x) or x == "…") else pd.to_datetime(x).strftime("%Y-%m-%d"),
                     'opening_stock': lambda x: "…" if (pd.isna(x) or x == "…") else f"{int(x)}",
                     'selling_price': lambda x: "…" if (pd.isna(x) or x == "…") else f"{float(x):.2f}",
                     'on_promotion':  lambda x: "…" if (pd.isna(x) or x == "…") else f"{int(x)}",
                     'quantity_sold': lambda x: "…" if (pd.isna(x) or x == "…") else f"{int(x)}",
                 }, na_rep="…")
                 .hide(axis="index")
           )


def display_engineered_slave_data(engineered_slave_data, n=3):
    
    head, tail = engineered_slave_data.head(n), engineered_slave_data.tail(n)
    sep = pd.DataFrame([["…"] * engineered_slave_data.shape[1]], 
                       columns=engineered_slave_data.columns, index=["…"])
    view = pd.concat([head, sep, tail])

    return (view.style
                 .format({
                     'date_stamp': lambda x: "…" if (pd.isna(x) or x == "…") 
                                              else pd.to_datetime(x).strftime("%Y-%m-%d"),
                     'sku': lambda x: "…" if (pd.isna(x) or x == "…") else str(x),
                     'trend': lambda x: "…" if (pd.isna(x) or x == "…") else f"{x:.2f}",
                     'weekly_seasonality': lambda x: "…" if (pd.isna(x) or x == "…") else f"{x:.2f}",
                     'monthly_seasonality': lambda x: "…" if (pd.isna(x) or x == "…") else f"{x:.2f}",
                     'quarterly_seasonality': lambda x: "…" if (pd.isna(x) or x == "…") else f"{x:.2f}",
                     'yearly_seasonality': lambda x: "…" if (pd.isna(x) or x == "…") else f"{x:.2f}",
                     'opening_stock': lambda x: "…" if (pd.isna(x) or x == "…") else f"{int(x)}",
                     'selling_price': lambda x: "…" if (pd.isna(x) or x == "…") else f"{x:.2f}",
                     'quantity_sold': lambda x: "…" if (pd.isna(x) or x == "…") else f"{int(x)}",
                 }, na_rep="…")
                 .hide(axis="index")
           )


def perform_bootstrap_test(estimate, simulations):
    """
    The method calculates a two-sided percentile p-value.
    See footnotes in https://journals.sagepub.com/doi/full/10.1177/2515245920911881

    This implementation is retrieved from the DoWhy causal refutation module:
    https://github.com/py-why/dowhy/blob/main/dowhy/causal_refuter.py#L148
    """
    
    half_p_value = np.mean([(x > estimate) + 0.5 * (x == estimate) for x in simulations])
    return 2 * min(half_p_value, 1 - half_p_value)

