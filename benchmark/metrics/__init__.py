"""Classification, survival, and segmentation metrics used by training."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from math import erfc, sqrt
import numpy as np
EPS = 1e-08
def _safe_div(num: float, den: float) -> float:
    return float(num / (den + EPS))
def concordance_index(event_times: np.ndarray, risks: np.ndarray, censorships: np.ndarray) -> float:
    """C-index via sksurv (frozen prognosis protocol; MCAT-compatible tied-time handling)."""
    from sksurv.metrics import concordance_index_censored
    event_times = np.asarray(event_times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    event_observed = np.asarray(censorships, dtype=int) == 0
    if event_times.size < 2:
        return 0.0
    else:
        result, _, _, _, _ = concordance_index_censored(event_observed.astype(bool), event_times, risks, tied_tol=1e-08)
        return float(result) if np.isfinite(result) else 0.0
def concordance_index_bootstrap_ci(event_times: np.ndarray, risks: np.ndarray, censorships: np.ndarray, *, n_bootstraps: int=1000, confidence_level: float=0.95, seed: int=42) -> dict[str, float]:
    """Patient-level bootstrap CI for survival C-index."""
    event_times = np.asarray(event_times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    censorships = np.asarray(censorships, dtype=float)
    n = int(len(event_times))
    if n < 2 or n_bootstraps <= 0:
        point = concordance_index(event_times, risks, censorships)
        return {'c_index_ci_low': point, 'c_index_ci_high': point, 'c_index_bootstrap_n': 0.0, 'c_index_bootstrap_level': float(confidence_level)}
    else:
        rng = np.random.default_rng(int(seed))
        samples = []
        for _ in range(int(n_bootstraps)):
            indices = rng.integers(0, n, size=n)
            value = concordance_index(event_times[indices], risks[indices], censorships[indices])
            if np.isfinite(value):
                samples.append(float(value))
        if not samples:
            point = concordance_index(event_times, risks, censorships)
            return {'c_index_ci_low': point, 'c_index_ci_high': point, 'c_index_bootstrap_n': 0.0, 'c_index_bootstrap_level': float(confidence_level)}
        else:
            alpha = 1.0 - float(confidence_level)
            low, high = np.quantile(np.asarray(samples, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
            return {'c_index_ci_low': float(low), 'c_index_ci_high': float(high), 'c_index_bootstrap_n': float(len(samples)), 'c_index_bootstrap_level': float(confidence_level)}
def logrank_by_median_risk(event_times: np.ndarray, risks: np.ndarray, censorships: np.ndarray, *, risk_threshold: float | None=None) -> dict[str, float]:
    """Two-group log-rank test after splitting patients by risk threshold."""
    # ***<module>.logrank_by_median_risk: Failure: Different control flow
    event_times = np.asarray(event_times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    observed = 1 - np.asarray(censorships).astype(int)
    if len(event_times) < 2 or np.all(risks == risks[0]):
        threshold = float(risks[0]) if len(risks) else 0.0
        return {'logrank_chi2': 0.0, 'logrank_p': 1.0, 'risk_median': threshold, 'risk_threshold': threshold, 'risk_group_low_n': float(len(risks)), 'risk_group_high_n': 0.0}
    else:
        threshold = float(np.median(risks) if risk_threshold is None else risk_threshold)
        high_risk = risks > threshold
        if risk_threshold is None and (not np.any(high_risk) or np.all(high_risk)):
                high_risk = risks >= threshold
        if np.any(high_risk) and np.all(high_risk):
            return {'logrank_chi2': 0.0, 'logrank_p': 1.0, 'risk_median': threshold, 'risk_threshold': threshold, 'risk_group_low_n': float(np.sum(~high_risk)), 'risk_group_high_n': float(np.sum(high_risk))}
        else:
            observed_minus_expected = 0.0
            variance = 0.0
            for time in np.unique(event_times[observed == 1]):
                at_risk = event_times >= time
                n_total = float(np.sum(at_risk))
                n_high = float(np.sum(at_risk & high_risk))
                d_total = float(np.sum((event_times == time) & (observed == 1)))
                d_high = float(np.sum((event_times == time) & (observed == 1) & high_risk))
                if n_total <= 1.0:
                    continue
                else:
                    expected_high = d_total * n_high / n_total
                    var_high = n_high * (n_total - n_high) * d_total * (n_total - d_total) / (n_total * n_total * (n_total - 1.0) + EPS)
                    observed_minus_expected += d_high - expected_high
                    variance += var_high
            chi2 = float(observed_minus_expected * observed_minus_expected / (variance + EPS))
            p_value = float(erfc(sqrt(max(chi2, 0.0) / 2.0)))
            return {'logrank_chi2': chi2, 'logrank_p': p_value, 'risk_median': threshold, 'risk_threshold': threshold, 'risk_group_low_n': float(np.sum(~high_risk)), 'risk_group_high_n': float(np.sum(high_risk))}
def kaplan_meier_by_risk_group(event_times: np.ndarray, risks: np.ndarray, censorships: np.ndarray, *, risk_threshold: float) -> list[dict[str, float | str]]:
    """Kaplan-Meier step data for low/high predicted-risk groups."""
    event_times = np.asarray(event_times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    observed = 1 - np.asarray(censorships).astype(int)
    high_risk = risks > float(risk_threshold)
    rows = []
    for group_name, mask in [('low', ~high_risk), ('high', high_risk)]:
        group_times = event_times[mask]
        group_observed = observed[mask]
        survival = 1.0
        rows.append({'risk_group': group_name, 'time': 0.0, 'survival_probability': 1.0, 'at_risk': float(len(group_times)), 'events': 0.0, 'censored': 0.0})
        if len(group_times) == 0:
            continue
        else:
            for time in np.unique(group_times):
                at_risk = float(np.sum(group_times >= time))
                events = float(np.sum((group_times == time) & (group_observed == 1)))
                censored = float(np.sum((group_times == time) & (group_observed == 0)))
                if at_risk > 0.0:
                    survival *= 1.0 - events / at_risk
                rows.append({'risk_group': group_name, 'time': float(time), 'survival_probability': float(survival), 'at_risk': at_risk, 'events': events, 'censored': censored})
    return rows
def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC for binary labels without external dependencies."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    total = len(positives) * len(negatives)
    if total == 0:
        return 0.0
    else:
        concordant = 0.0
        for pos in positives:
            concordant += np.sum(pos > negatives)
            concordant += 0.5 * np.sum(pos == negatives)
        return float(concordant / total)
def binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision for binary labels, grouping tied scores."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0:
        return 0.0
    else:
        order = np.argsort(-scores, kind='stable')
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        threshold_indices = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
        true_positives = np.cumsum(sorted_labels)[threshold_indices]
        predicted_positives = threshold_indices + 1
        precision = true_positives / predicted_positives
        recall = true_positives / positive_count
        recall_increase = np.diff(np.r_[0.0, recall])
        return float(np.sum(recall_increase * precision))
def classification_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, num_classes: int) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    accuracy = float(np.mean(labels == predictions)) if labels.size else 0.0
    f1s, precisions, recalls = ([], [], [])
    out = {'accuracy': accuracy, 'accuracy_overall': accuracy}
    for cls in range(num_classes):
        tp = np.sum((labels == cls) & (predictions == cls))
        fp = np.sum((labels != cls) & (predictions == cls))
        fn = np.sum((labels == cls) & (predictions != cls))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = float(np.sum(labels == cls))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        out[f'class_{cls}/precision'] = precision
        out[f'class_{cls}/recall'] = recall
        out[f'class_{cls}/f1'] = f1
        out[f'class_{cls}/support'] = support
    out['precision_macro'] = float(np.mean(precisions)) if precisions else 0.0
    out['recall_macro'] = float(np.mean(recalls)) if recalls else 0.0
    out['f1_macro'] = float(np.mean(f1s)) if f1s else 0.0
    out['balanced_accuracy'] = float(np.mean(recalls)) if recalls else 0.0
    if num_classes == 2 and probabilities.ndim == 2 and (probabilities.shape[1] > 1):
        positive_probs = probabilities[:, 1]
        out['auc'] = binary_auc(labels, positive_probs)
        out['auprc'] = binary_auprc(labels, positive_probs)
        out['sensitivity'] = float(recalls[1])
        out['specificity'] = float(recalls[0])
        out['brier_score'] = float(np.mean((positive_probs - labels) ** 2))
        confidence = np.maximum(positive_probs, 1.0 - positive_probs)
        correctness = (predictions == labels).astype(float)
        ece = 0.0
        for lower in np.linspace(0.0, 1.0, 11)[:(-1)]:
            upper = lower + 0.1
            mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
            if np.any(mask):
                ece += float(np.mean(mask)) * abs(float(np.mean(correctness[mask])) - float(np.mean(confidence[mask])))
        out['calibration_error'] = float(ece)
    else:
        out['auc'] = 0.0
        out['auprc'] = 0.0
    return out
def predictions_from_threshold(positive_probs: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions from positive-class probability and a decision threshold."""
    return (np.asarray(positive_probs, dtype=float) >= float(threshold)).astype(int)
def tune_binary_threshold(labels: np.ndarray, positive_probs: np.ndarray, *, metric: str='f1_macro', grid: int=91, t_min: float=0.05, t_max: float=0.95) -> tuple[float, float]:
    """Pick a decision threshold on validation probabilities.\n\n    Supported ``metric`` values: ``f1_macro``, ``balanced_accuracy``, ``youden``.\n    """
    labels = np.asarray(labels).astype(int)
    positive_probs = np.asarray(positive_probs, dtype=float)
    if labels.size == 0:
        return (0.5, 0.0)
    else:
        thresholds = np.linspace(float(t_min), float(t_max), int(grid))
        best_threshold = 0.5
        best_score = -float('inf')
        for threshold in thresholds:
            predictions = predictions_from_threshold(positive_probs, threshold)
            if metric == 'balanced_accuracy':
                tp = np.sum((labels == 1) & (predictions == 1))
                tn = np.sum((labels == 0) & (predictions == 0))
                fp = np.sum((labels == 0) & (predictions == 1))
                fn = np.sum((labels == 1) & (predictions == 0))
                score = 0.5 * (_safe_div(tp, tp + fn) + _safe_div(tn, tn + fp))
            else:
                if metric == 'youden':
                    tp = np.sum((labels == 1) & (predictions == 1))
                    tn = np.sum((labels == 0) & (predictions == 0))
                    fp = np.sum((labels == 0) & (predictions == 1))
                    fn = np.sum((labels == 1) & (predictions == 0))
                    score = _safe_div(tp, tp + fn) + _safe_div(tn, tn + fp) - 1.0
                else:
                    probs = np.stack([1.0 - positive_probs, positive_probs], axis=1)
                    score = classification_metrics(labels, predictions, probs, num_classes=2)[metric]
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)
        return (best_threshold, best_score)
def linear_probe_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, num_classes: int, *, prefix: str='lin_', class_names: list[str] | None=None) -> dict[str, float | dict]:
    """UNI-style patch-classification metrics for linear probing."""
    try:
        from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, roc_auc_score
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('linear_probe_metrics requires scikit-learn. Install scikit-learn or run a non-linear-probe evaluation.') from exc
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    class_ids = list(range(int(num_classes)))
    report = classification_report(labels, predictions, labels=class_ids, target_names=class_names if class_names else None, output_dict=True, zero_division=0)
    metrics = {f'{prefix}acc': float(accuracy_score(labels, predictions)), f'{prefix}bacc': float(balanced_accuracy_score(labels, predictions)), f'{prefix}macro_f1': float(report['macro avg']['f1-score']), f'{prefix}weighted_f1': float(report['weighted avg']['f1-score']), f'{prefix}report': report}
    try:
        if int(num_classes) == 2:
            score = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
            metrics[f'{prefix}auroc'] = float(roc_auc_score(labels, score))
        else:
            metrics[f'{prefix}auroc'] = float(roc_auc_score(labels, probabilities, multi_class='ovo', average='macro'))
    except ValueError:
        metrics[f'{prefix}auroc'] = float('nan')
    try:
        if int(num_classes) == 2:
            score = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
            metrics[f'{prefix}auprc'] = float(average_precision_score(labels, score))
        else:
            metrics[f'{prefix}auprc'] = float(
                average_precision_score(labels, probabilities, average='macro')
            )
    except ValueError:
        metrics[f'{prefix}auprc'] = float('nan')
    return metrics
def _extract_numbers_from_string(value: str) -> list[str]:
    import re
    return re.findall('\\d+\\.?\\d*', value)
def _pfmb_binary_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str, class_id: int | None) -> float:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    labels = [class_id] if class_id is not None else None
    if metric_name == 'precision':
        return float(precision_score(y_true, y_pred, labels=labels, average='micro', zero_division=0))
    else:
        if metric_name == 'recall':
            return float(recall_score(y_true, y_pred, labels=labels, average='micro', zero_division=0))
        else:
            if metric_name == 'f1':
                return float(f1_score(y_true, y_pred, labels=labels, average='micro', zero_division=0))
            else:
                if metric_name == 'accuracy':
                    if class_id is not None:
                        mask = y_true == class_id
                        if not np.any(mask):
                            return 0.0
                        else:
                            return float(accuracy_score(y_true[mask], y_pred[mask]))
                    else:
                        return float(accuracy_score(y_true, y_pred))
                else:
                    raise ValueError(f'Unsupported PFMB metric \'{metric_name}\'')
def get_instance_level_labels(pred_probs: np.ndarray, semantic_gt: np.ndarray, instance_gt: np.ndarray, label_dict: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract PFMB-style per-instance GT and predicted class labels."""
    import pandas as pd
    from skimage.measure import regionprops_table
    instance_gt = np.asarray(instance_gt)
    semantic_gt = np.asarray(semantic_gt)
    if instance_gt.ndim != 2 or semantic_gt.ndim != 2:
        raise ValueError('instance_gt and semantic_gt must be 2D arrays')
    else:
        props_gt = regionprops_table(instance_gt, intensity_image=semantic_gt.astype(np.float64), properties=('label', 'intensity_mean'))
        props_gt = pd.DataFrame(props_gt)
        if props_gt.empty:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        else:
            props_gt = props_gt.rename(columns={'intensity_mean': 'groundtruth'})
            props_gt['groundtruth'] = props_gt['groundtruth'].astype(int)
            semantic_pred = np.moveaxis(np.asarray(pred_probs), 0, (-1))
            props_pred = regionprops_table(instance_gt, intensity_image=semantic_pred, properties=('label', 'intensity_mean'))
            props_pred = pd.DataFrame(props_pred)
            label_dict_rev = {name: int(cls_id) for cls_id, name in label_dict.items()}
            rename_dict = {}
            for column in props_pred.columns:
                if column == 'label':
                    continue
                channel_ids = _extract_numbers_from_string(column)
                if not channel_ids:
                    continue
                class_id = int(float(channel_ids[0]))
                if class_id in label_dict:
                    rename_dict[column] = label_dict[class_id]
            props_pred = props_pred.rename(columns=rename_dict)
            class_cols = [col for col in props_pred.columns if col != 'label']
            if not class_cols:
                return (props_gt['groundtruth'].to_numpy(dtype=np.int64), np.zeros(len(props_gt), dtype=np.int64))
            else:
                props_pred['pred_class_name'] = props_pred[class_cols].idxmax(axis=1)
                props_pred['pred_class'] = props_pred['pred_class_name'].map(label_dict_rev).fillna(0).astype(int)
                pred_df = pd.merge(props_gt[['label', 'groundtruth']], props_pred[['label', 'pred_class']], on='label', how='inner')
                return (pred_df['groundtruth'].to_numpy(dtype=np.int64), pred_df['pred_class'].to_numpy(dtype=np.int64))
def get_instance_level_probabilities(pred_probs: np.ndarray, semantic_gt: np.ndarray, instance_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return GT class ids and mean softmax vectors per GT instance."""
    import pandas as pd
    from skimage.measure import regionprops_table
    instance_gt = np.asarray(instance_gt)
    semantic_gt = np.asarray(semantic_gt)
    pred_probs = np.asarray(pred_probs)
    if pred_probs.ndim != 3:
        raise ValueError('pred_probs must be (C,H,W)')
    else:
        props_gt = regionprops_table(instance_gt, intensity_image=semantic_gt.astype(np.float64), properties=('label', 'intensity_mean'))
        props_gt = pd.DataFrame(props_gt)
        if props_gt.empty:
            return (np.array([], dtype=np.int64), np.zeros((0, pred_probs.shape[0]), dtype=np.float64))
        else:
            props_gt = props_gt.rename(columns={'intensity_mean': 'groundtruth'})
            props_gt['groundtruth'] = props_gt['groundtruth'].astype(int)
            semantic_pred = np.moveaxis(pred_probs, 0, (-1))
            props_pred = regionprops_table(instance_gt, intensity_image=semantic_pred.astype(np.float64), properties=('label', 'intensity_mean'))
            props_pred = pd.DataFrame(props_pred)
            prob_cols = [col for col in props_pred.columns if col != 'label']
            if not prob_cols:
                return (props_gt['groundtruth'].to_numpy(dtype=np.int64), np.zeros((len(props_gt), pred_probs.shape[0]), dtype=np.float64))
            else:
                merged = pd.merge(props_gt[['label', 'groundtruth']], props_pred[['label', *prob_cols]], on='label', how='inner')
                prob_matrix = merged[prob_cols].to_numpy(dtype=np.float64)
                if prob_matrix.shape[1] != pred_probs.shape[0]:
                    raise ValueError(f'Expected {pred_probs.shape[0]} class probability columns, got {prob_matrix.shape[1]}')
                else:
                    return (merged['groundtruth'].to_numpy(dtype=np.int64), prob_matrix)
def _pearson_from_sums(n: float, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    if n <= 1.0:
        return 0.0
    else:
        num = n * sxy - sx * sy
        den_x = n * sxx - sx * sx
        den_y = n * syy - sy * sy
        if den_x <= 0.0 or den_y <= 0.0:
            return 0.0
        else:
            return float(num / (np.sqrt(den_x * den_y) + EPS))
@dataclass
class PixelPearsonAccumulator:
    """MIPHEI-style pixel Pearson r between class probabilities and binary GT masks."""
    num_classes: int
    exclude_background: bool = True
    def __post_init__(self) -> None:
        self._stats = {cls_id: {'n': 0.0, 'sx': 0.0, 'sy': 0.0, 'sxx': 0.0, 'syy': 0.0, 'sxy': 0.0} for cls_id in range(self.num_classes)}
    def update(self, pred_probs: np.ndarray, target: np.ndarray) -> None:
        pred_probs = np.asarray(pred_probs, dtype=np.float64)
        target = np.asarray(target)
        if pred_probs.ndim != 3:
            raise ValueError('pred_probs must be (C,H,W)')
        else:
            for cls_id in range(self.num_classes):
                if self.exclude_background and cls_id == 0:
                        continue
                pred_flat = pred_probs[cls_id].reshape((-1)) * 255.0
                target_flat = (target == cls_id).reshape((-1)).astype(np.float64) * 255.0
                stats = self._stats[cls_id]
                stats['n'] += float(pred_flat.size)
                stats['sx'] += float(pred_flat.sum())
                stats['sy'] += float(target_flat.sum())
                stats['sxx'] += float(np.square(pred_flat).sum())
                stats['syy'] += float(np.square(target_flat).sum())
                stats['sxy'] += float(np.dot(pred_flat, target_flat))
    def compute(self, label_dict: dict[int, str] | None=None) -> dict[str, float]:
        per_class = {}
        for cls_id, stats in self._stats.items():
            if self.exclude_background and cls_id == 0:
                    continue
            per_class[cls_id] = _pearson_from_sums(stats['n'], stats['sx'], stats['sy'], stats['sxx'], stats['syy'], stats['sxy'])
        present = [cls_id for cls_id, value in per_class.items() if value != 0.0 or self._stats[cls_id]['sy'] > 0]
        eval_ids = present if present else list(per_class)
        macro = float(np.mean([per_class[cls_id] for cls_id in eval_ids])) if eval_ids else 0.0
        out = {'pixel_pearson_r': macro, 'pearson_r_macro': macro, 'pixel_pearson_class_count': float(len(eval_ids))}
        for cls_id, value in per_class.items():
            class_name = (label_dict or {}).get(cls_id, f'class_{cls_id}')
            out[f'class_{cls_id}/pearson_r'] = value
            out[f'{class_name}/pearson_r'] = value
        return out
@dataclass
class CellAUPRCAccumulator:
    """MIPHEI-style cell AUPRC using per-cell mean softmax scores (one-vs-rest per class)."""
    label_dict: dict[int, str]
    num_classes: int
    exclude_background: bool = True
    def __post_init__(self) -> None:
        self._gt_classes = []
        self._mean_probs = []
    def update(self, pred_probs: np.ndarray, semantic_gt: np.ndarray, instance_gt: np.ndarray) -> None:
        gt_classes, mean_probs = get_instance_level_probabilities(pred_probs, semantic_gt, instance_gt)
        if gt_classes.size:
            self._gt_classes.extend((int(v) for v in gt_classes))
            self._mean_probs.extend(mean_probs)
    def compute(self) -> dict[str, float]:
        from sklearn.metrics import average_precision_score
        if not self._gt_classes:
            return {'cell_auprc_macro': 0.0, 'auprc_macro': 0.0, 'cell_auprc_class_count': 0.0, 'cell_auprc_instance_count': 0.0}
        else:
            y_true = np.asarray(self._gt_classes, dtype=np.int64)
            y_score = np.vstack(self._mean_probs).astype(np.float64)
            class_ids = np.arange(self.num_classes)
            if self.exclude_background:
                class_ids = class_ids[class_ids != 0]
            ap_scores = {}
            for cls_id in class_ids:
                labels = (y_true == cls_id).astype(np.int64)
                if labels.sum() == 0 or labels.sum() == len(labels):
                    continue
                else:
                    ap_scores[int(cls_id)] = float(average_precision_score(labels, y_score[:, int(cls_id)]))
            eval_ids = list(ap_scores)
            macro = float(np.mean(list(ap_scores.values()))) if ap_scores else 0.0
            out = {'cell_auprc_macro': macro, 'auprc_macro': macro, 'cell_auprc_class_count': float(len(eval_ids)), 'cell_auprc_instance_count': float(len(y_true))}
            for cls_id, value in ap_scores.items():
                class_name = self.label_dict.get(cls_id, f'class_{cls_id}')
                out[f'class_{cls_id}/auprc'] = value
                out[f'{class_name}/auprc'] = value
            return out
@dataclass
class InstanceLevelMetricAccumulator:
    """PFMB-faithful instance-level metrics using GT instance masks."""
    label_dict: dict[int, str]
    def __post_init__(self) -> None:
        self._y_true = []
        self._y_pred = []
    def update(self, pred_probs: np.ndarray, semantic_gt: np.ndarray, instance_gt: np.ndarray) -> None:
        y_true, y_pred = get_instance_level_labels(pred_probs, semantic_gt, instance_gt, self.label_dict)
        if y_true.size:
            self._y_true.extend((int(v) for v in y_true))
            self._y_pred.extend((int(v) for v in y_pred))
    def compute(self) -> dict[str, float]:
        if not self._y_true:
            return {'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_score_macro': 0.0, 'accuracy_macro': 0.0, 'precision_micro': 0.0, 'recall_micro': 0.0, 'f1_score_micro': 0.0, 'accuracy_micro': 0.0, 'instance_count': 0.0}
        else:
            y_true = np.asarray(self._y_true, dtype=np.int64)
            y_pred = np.asarray(self._y_pred, dtype=np.int64)
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            unique_classes = unique_classes[unique_classes != 0]
            precision_scores = {}
            recall_scores = {}
            f1_scores = {}
            accuracy_scores = {}
            out = {'instance_count': float(len(y_true))}
            for cls in unique_classes:
                class_name = self.label_dict.get(int(cls), f'class_{int(cls)}')
                precision_scores[class_name] = _pfmb_binary_metric(y_true, y_pred, 'precision', int(cls))
                recall_scores[class_name] = _pfmb_binary_metric(y_true, y_pred, 'recall', int(cls))
                f1_scores[class_name] = _pfmb_binary_metric(y_true, y_pred, 'f1', int(cls))
                accuracy_scores[class_name] = _pfmb_binary_metric(y_true, y_pred, 'accuracy', int(cls))
                out[f'{class_name}/precision'] = precision_scores[class_name]
                out[f'{class_name}/recall'] = recall_scores[class_name]
                out[f'{class_name}/f1_score'] = f1_scores[class_name]
                out[f'{class_name}/accuracy'] = accuracy_scores[class_name]
            out['precision_macro'] = float(np.mean(list(precision_scores.values()))) if precision_scores else 0.0
            out['recall_macro'] = float(np.mean(list(recall_scores.values()))) if recall_scores else 0.0
            out['f1_score_macro'] = float(np.mean(list(f1_scores.values()))) if f1_scores else 0.0
            out['accuracy_macro'] = float(np.mean(list(accuracy_scores.values()))) if accuracy_scores else 0.0
            out['precision_micro'] = _pfmb_binary_metric(y_true, y_pred, 'precision', None)
            out['recall_micro'] = _pfmb_binary_metric(y_true, y_pred, 'recall', None)
            out['f1_score_micro'] = _pfmb_binary_metric(y_true, y_pred, 'f1', None)
            out['accuracy_micro'] = _pfmb_binary_metric(y_true, y_pred, 'accuracy', None)
            return out
@dataclass
class SegmentationMetricAccumulator:
    num_classes: int
    ignore_index: int | None = None
    exclude_background: bool = True
    def __post_init__(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(pred).reshape((-1))
        target = np.asarray(target).reshape((-1))
        valid = (target >= 0) & (target < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)
        if self.ignore_index is not None:
            valid &= target != self.ignore_index
        encoded = self.num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
        self.confusion += np.bincount(encoded, minlength=self.num_classes * self.num_classes).reshape(self.num_classes, self.num_classes)
    def compute(self) -> dict[str, float]:
        tp = np.diag(self.confusion).astype(np.float64)
        support = self.confusion.sum(axis=1).astype(np.float64)
        predicted = self.confusion.sum(axis=0).astype(np.float64)
        fp = predicted - tp
        fn = support - tp
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)
        dice = 2 * tp / (2 * tp + fp + fn + EPS)
        iou = tp / (tp + fp + fn + EPS)
        class_accuracy = tp / (support + EPS)
        class_ids = np.arange(self.num_classes)
        eval_ids = class_ids[class_ids != 0] if self.exclude_background else class_ids
        present = support[eval_ids] > 0
        avg_ids = eval_ids[present] if np.any(present) else eval_ids
        if self.num_classes > 1:
            foreground_tp = self.confusion[1:, 1:].sum()
            foreground_fp = self.confusion[0, 1:].sum()
            foreground_fn = self.confusion[1:, 0].sum()
            foreground_iou = _safe_div(foreground_tp, foreground_tp + foreground_fp + foreground_fn)
        else:
            foreground_iou = 0.0
        out = {
            'accuracy_overall': _safe_div(tp[eval_ids].sum(), support[eval_ids].sum()),
            'precision_macro': float(np.mean(precision[avg_ids])) if len(avg_ids) else 0.0,
            'recall_macro': float(np.mean(recall[avg_ids])) if len(avg_ids) else 0.0,
            'f1_macro': float(np.mean(f1[avg_ids])) if len(avg_ids) else 0.0,
            'dice_macro': float(np.mean(dice[avg_ids])) if len(avg_ids) else 0.0,
            'iou_macro': float(np.mean(iou[avg_ids])) if len(avg_ids) else 0.0,
            'foreground_iou': foreground_iou,
        }
        tp_micro = tp[avg_ids].sum()
        fp_micro = fp[avg_ids].sum()
        fn_micro = fn[avg_ids].sum()
        precision_micro = _safe_div(tp_micro, tp_micro + fp_micro)
        recall_micro = _safe_div(tp_micro, tp_micro + fn_micro)
        out.update({'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': _safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro), 'dice_micro': _safe_div(2 * tp_micro, 2 * tp_micro + fp_micro + fn_micro), 'iou_micro': _safe_div(tp_micro, tp_micro + fp_micro + fn_micro)})
        for cls in eval_ids:
            prefix = f'class_{int(cls)}'
            out[f'{prefix}/precision'] = float(precision[cls])
            out[f'{prefix}/recall'] = float(recall[cls])
            out[f'{prefix}/f1'] = float(f1[cls])
            out[f'{prefix}/dice'] = float(dice[cls])
            out[f'{prefix}/iou'] = float(iou[cls])
            out[f'{prefix}/accuracy'] = float(class_accuracy[cls])
            out[f'{prefix}/support'] = float(support[cls])
        total_predicted = float(predicted.sum())
        for cls in class_ids:
            out[f'class_{int(cls)}/predicted_fraction'] = _safe_div(predicted[cls], total_predicted)
        return out
def _majority_class(values: np.ndarray) -> int:
    values = values[values > 0]
    if values.size == 0:
        return 0
    else:
        counts = np.bincount(values.astype(np.int64))
        return int(np.argmax(counts))
def connected_components(mask: np.ndarray) -> np.ndarray:
    """4-connected component labels for a binary 2D mask."""
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    height, width = mask.shape
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x] != 0:
                continue
            current += 1
            stack = [(start_y, start_x)]
            labels[start_y, start_x] = current
            while stack:
                y, x = stack.pop()
                for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels
def _pred_instances(pred_sem: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    instances = []
    next_id = 1
    for cls in sorted((int(x) for x in np.unique(pred_sem) if int(x) != 0)):
        labels = connected_components(pred_sem == cls)
        for label_id in range(1, int(labels.max()) + 1):
            mask = labels == label_id
            if mask.any():
                instances.append((next_id, cls, mask))
                next_id += 1
    return instances
def _gt_instances(gt_sem: np.ndarray, gt_inst: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    instances = []
    for inst_id in sorted((int(x) for x in np.unique(gt_inst) if int(x) != 0)):
        mask = gt_inst == inst_id
        cls = _majority_class(gt_sem[mask])
        if cls != 0:
            instances.append((inst_id, cls, mask))
    return instances
class InstanceIoUMatcher:
    """Greedy IoU matcher for connected predicted cells against GT instances."""
    def __init__(self, num_classes: int, iou_threshold: float=0.5):
        self.num_classes = num_classes
        self.iou_threshold = float(iou_threshold)
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.ious = []
    def update(self, pred_sem: np.ndarray, gt_sem: np.ndarray, gt_inst: np.ndarray) -> None:
        pred_instances = _pred_instances(pred_sem)
        gt_instances = _gt_instances(gt_sem, gt_inst)
        candidates = []
        for pred_idx, (_, pred_cls, pred_mask) in enumerate(pred_instances):
            for gt_idx, (_, gt_cls, gt_mask) in enumerate(gt_instances):
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                if intersection == 0:
                    continue
                else:
                    union = np.logical_or(pred_mask, gt_mask).sum()
                    iou = float(intersection / (union + EPS))
                    if iou >= self.iou_threshold:
                        candidates.append((iou, pred_idx, gt_idx, pred_cls, gt_cls))
        matched_pred = set()
        matched_gt = set()
        for iou, pred_idx, gt_idx, pred_cls, gt_cls in sorted(candidates, reverse=True):
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            else:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
                self.ious.append(iou)
                if pred_cls == gt_cls:
                    self.tp[gt_cls] += 1
                else:
                    self.fp[pred_cls] += 1
                    self.fn[gt_cls] += 1
        for pred_idx, (_, pred_cls, _) in enumerate(pred_instances):
            if pred_idx not in matched_pred:
                self.fp[pred_cls] += 1
        for gt_idx, (_, gt_cls, _) in enumerate(gt_instances):
            if gt_idx not in matched_gt:
                self.fn[gt_cls] += 1
    def compute(self) -> dict[str, float]:
        classes = range(1, self.num_classes)
        out = {'instance_iou_match/mean_iou': float(np.mean(self.ious)) if self.ious else 0.0}
        f1s = []
        tp_total = fp_total = fn_total = 0
        for cls in classes:
            tp, fp, fn = (self.tp[cls], self.fp[cls], self.fn[cls])
            tp_total += tp
            fp_total += fp
            fn_total += fn
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * precision * recall, precision + recall)
            f1s.append(f1)
            prefix = f'instance_iou_match/class_{cls}'
            out[f'{prefix}/precision'] = precision
            out[f'{prefix}/recall'] = recall
            out[f'{prefix}/f1'] = f1
        precision_micro = _safe_div(tp_total, tp_total + fp_total)
        recall_micro = _safe_div(tp_total, tp_total + fn_total)
        out['instance_iou_match/precision_micro'] = precision_micro
        out['instance_iou_match/recall_micro'] = recall_micro
        out['instance_iou_match/f1_micro'] = _safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro)
        out['instance_iou_match/f1_macro'] = float(np.mean(f1s)) if f1s else 0.0
        return out
def compute_extended_pixel_metrics(confusion: np.ndarray, *, exclude_background: bool=True) -> dict[str, float]:
    """Additional mainstream pixel metrics derived from a confusion matrix."""
    confusion = np.asarray(confusion, dtype=np.float64)
    num_classes = confusion.shape[0]
    tp = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    fp = predicted - tp
    fn = support - tp
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    specificity_vals = []
    for cls in range(num_classes):
        tp_c = tp[cls]
        fp_c = fp[cls]
        fn_c = fn[cls]
        tn_c = confusion.sum() - tp_c - fp_c - fn_c
        specificity_vals.append(tn_c / (tn_c + fp_c + EPS))
    specificity = np.asarray(specificity_vals, dtype=np.float64)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)
    dice = 2 * tp / (2 * tp + fp + fn + EPS)
    class_accuracy = tp / (support + EPS)
    class_ids = np.arange(num_classes)
    eval_ids = class_ids[class_ids != 0] if exclude_background else class_ids
    present = support[eval_ids] > 0
    avg_ids = eval_ids[present] if np.any(present) else eval_ids
    mcc_vals = []
    for cls in avg_ids:
        tp_c = tp[cls]
        fp_c = fp[cls]
        fn_c = fn[cls]
        tn_c = confusion.sum() - tp_c - fp_c - fn_c
        denom = np.sqrt((tp_c + fp_c) * (tp_c + fn_c) * (tn_c + fp_c) * (tn_c + fn_c))
        mcc_vals.append(float((tp_c * tn_c - fp_c * fn_c) / (denom + EPS)))
    total = confusion.sum()
    po = _safe_div(np.trace(confusion), total)
    pe = 0.0
    for cls in range(num_classes):
        pe += _safe_div(confusion[:, cls].sum() * confusion[cls, :].sum(), total * total)
    kappa = _safe_div(po - pe, 1.0 - pe)
    weighted_iou = 0.0
    weight_sum = 0.0
    for cls in avg_ids:
        weight = float(support[cls])
        weighted_iou += weight * float(iou[cls])
        weight_sum += weight
    out = {
        'balanced_accuracy': float(np.mean(recall[avg_ids])) if len(avg_ids) else 0.0,
        'mean_class_accuracy': float(np.mean(class_accuracy[avg_ids])) if len(avg_ids) else 0.0,
        'specificity_macro': float(np.mean(specificity[avg_ids])) if len(avg_ids) else 0.0,
        'mcc_macro': float(np.mean(mcc_vals)) if mcc_vals else 0.0,
        'cohen_kappa': kappa,
        'fw_iou': _safe_div(weighted_iou, weight_sum),
        'macro_dice': float(np.mean(dice[avg_ids])) if len(avg_ids) else 0.0,
        'macro_jaccard': float(np.mean(iou[avg_ids])) if len(avg_ids) else 0.0,
        'foreground_mean_recall': float(np.mean(recall[avg_ids])) if len(avg_ids) else 0.0,
        'foreground_mean_precision': float(np.mean(precision[avg_ids])) if len(avg_ids) else 0.0,
    }
    return out
class AggregatedJaccardAccumulator:
    """AJI-style aggregated Jaccard for instance masks (common in cell/nuclei papers)."""
    def __init__(self) -> None:
        self.matched_iou_sum = 0.0
        self.gt_count = 0
        self.unmatched_pred = 0
    def update(self, pred_sem: np.ndarray, gt_sem: np.ndarray, gt_inst: np.ndarray) -> None:
        pred_instances = _pred_instances(pred_sem)
        gt_instances = _gt_instances(gt_sem, gt_inst)
        self.gt_count += len(gt_instances)
        used_pred = set()
        for _, _, gt_mask in gt_instances:
            best_iou = 0.0
            best_idx = (-1)
            for pred_idx, (_, _, pred_mask) in enumerate(pred_instances):
                if pred_idx in used_pred:
                    continue
                else:
                    intersection = np.logical_and(pred_mask, gt_mask).sum()
                    if intersection == 0:
                        continue
                    else:
                        union = np.logical_or(pred_mask, gt_mask).sum()
                        iou = float(intersection / (union + EPS))
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = pred_idx
            if best_idx >= 0:
                used_pred.add(best_idx)
                self.matched_iou_sum += best_iou
        self.unmatched_pred += sum((1 for pred_idx in range(len(pred_instances)) if pred_idx not in used_pred))
    def compute(self) -> dict[str, float]:
        denom = self.gt_count + self.unmatched_pred
        return {'aji': _safe_div(self.matched_iou_sum, denom)}
PATHOCELL_MAINSTREAM_METRIC_KEYS = ('f1_macro', 'dice_macro', 'iou_macro', 'precision_macro', 'recall_macro', 'f1_micro', 'dice_micro', 'iou_micro', 'precision_micro', 'recall_micro', 'accuracy_overall', 'foreground_iou', 'balanced_accuracy', 'mean_class_accuracy', 'specificity_macro', 'mcc_macro', 'cohen_kappa', 'fw_iou', 'macro_dice', 'macro_jaccard', 'foreground_mean_recall', 'foreground_mean_precision', 'f1_score_macro', 'f1_score_micro', 'accuracy_macro', 'instance_iou_match/f1_macro', 'instance_iou_match/f1_micro', 'instance_iou_match/mean_iou', 'aji', 'cell_auprc_macro', 'auprc_macro', 'pixel_pearson_r', 'pearson_r_macro')
PFMB_INSTANCE_AGGREGATE_KEYS = ('f1_score_macro', 'precision_macro', 'recall_macro', 'accuracy_macro', 'f1_score_micro', 'precision_micro', 'recall_micro', 'accuracy_micro', 'instance_count', 'loss')
def uses_pfmb_instance_primary_metric(cfg) -> bool:
    """PathoCell cell semantic seg with PFMB instance-level checkpoint selection."""
    return getattr(cfg, 'task', None) == 'cell_semantic_segmentation' and str(getattr(cfg, 'training', {}).get('primary_metric', '')) == 'f1_score_macro'
def aggregate_pfmb_instance_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0)) for key in PFMB_INSTANCE_AGGREGATE_KEYS}
def format_pfmb_instance_validation_log(metrics: dict[str, float]) -> str:
    return f"valid_f1_score_macro={metrics.get('f1_score_macro', 0.0):.4f} valid_precision_macro={metrics.get('precision_macro', 0.0):.4f} valid_recall_macro={metrics.get('recall_macro', 0.0):.4f} instance_count={int(metrics.get('instance_count', 0.0))}"
