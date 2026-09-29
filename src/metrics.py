import numpy as np


def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))


def pearson_corr(pred, gt):
    if np.std(pred) < 1e-9 or np.std(gt) < 1e-9:
        return float("nan")
    return float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1])


def bias(pred, gt):
    return float(np.mean(pred - gt))


def median_abs_error(pred, gt):
    return float(np.median(np.abs(pred - gt)))


def error_percentiles(pred, gt, ps=(50, 75, 90, 95, 99)):
    err = np.abs(pred - gt)
    return {f"abs_err_p{p}": float(np.percentile(err, p)) for p in ps}


def all_metrics(pred, gt, prefix=""):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    out = {
        f"{prefix}rmse": rmse(pred, gt),
        f"{prefix}mae": mae(pred, gt),
        f"{prefix}corr": pearson_corr(pred, gt),
        f"{prefix}bias": bias(pred, gt),
        f"{prefix}median_abs_err": median_abs_error(pred, gt),
    }
    out.update({f"{prefix}{k}": v for k, v in error_percentiles(pred, gt).items()})
    return out


def fit_scale_shift(pred, gt):
    """Least-squares gt ~= a*pred + b. Returns (a, b)."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    A = np.vstack([pred, np.ones_like(pred)]).T
    (a, b), *_ = np.linalg.lstsq(A, gt, rcond=None)
    return float(a), float(b)
