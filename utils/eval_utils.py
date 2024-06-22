import numpy as np

from sklearn.metrics import precision_recall_fscore_support, \
                            roc_auc_score, average_precision_score, \
                            log_loss

def classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2):
    """Build a text report showing the main classification metrics.

    Read more in the :ref:`User Guide <classification_report>`.

    Args:
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.

        y_pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.

        labels : array-like of shape (n_labels,), default=None
            Optional list of label indices to include in the report.

        target_names : list of str of shape (n_labels,), default=None
            Optional display names matching the labels (same order).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        digits : int, default=2
            Number of digits for formatting output floating point values.

    Returns:
    -------
    report : str or dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::

            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }

        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy
        otherwise and would be the same for all metrics.

        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    """
    if labels is None:
        labels = np.unique(y_true)
    else:
        labels = np.asarray(labels)

    if target_names is not None and len(labels) != len(target_names):
        raise ValueError(
            "Number of classes, {0}, does not match size of "
            "target_names, {1}. Try specifying the labels "
            "parameter".format(len(labels), len(target_names))
        )

    if target_names is None:
        target_names = [str(l) for l in labels]

    headers = ["class", "support", "precision", "recall", "f1-score", "avg-precision", "roc-auc-score", "log-loss"]

    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight
    )
    rows = zip(target_names, s, p, r, f1)

    longest_last_line_heading = "accuracy (micro)"
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), digits)

    head_fmt = "{:>{width}s} |" + " {:<9} |" * (len(headers)-1)
    report = head_fmt.format(*headers, width=width) + "\n"

    len_report = len(report)
    report += '-'*len_report + "\n"

    row_fmt = "{:>{width}s} |" + " {:<9} |" + " {:<9.{digits}f} |" * 3 + " {:<13} |" * 2 + " {:<9} |" + "\n"
    for row in rows:
        report += row_fmt.format(*row, *tuple('-')*3, width=width, digits=digits)

    report += '-'*len_report + "\n"

    # compute all applicable averages
    average_options = ("micro", "macro", "weighted")
    for average in average_options:
        if average.startswith("micro"):
            line_heading = "accuracy (micro)"
        else:
            line_heading = average + " avg"

        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            sample_weight=sample_weight
        )

        roc_auc_s = roc_auc_score(y_true, y_pred)
        avg_p_ = average_precision_score(y_true, y_pred)
        log_loss_ = log_loss(y_true, y_pred)
        avg_metrics = [np.sum(s), avg_p, avg_r, avg_f1, avg_p_, roc_auc_s, log_loss_]

        row_fmt = "{:>{width}s} |" + " {:<9} |" + " {:<9.{digits}f} |" * 3 + " {:<13.{digits}f} |" * 2 + " {:<9.{digits}f} |" + "\n"
        report += row_fmt.format(line_heading, *avg_metrics, width=width, digits=digits)

    return report