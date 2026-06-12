function metrics = compute_fold_metrics(y_true, y_pred)
%COMPUTE_FOLD_METRICS Binary classification metrics (accuracy, precision, recall, F1).
%
%   metrics = compute_fold_metrics(y_true, y_pred)
%
%   Handles edge cases: empty inputs, single-class folds, mismatched sizes.

    y_true = y_true(:);
    y_pred = y_pred(:);

    if isempty(y_true) || isempty(y_pred) || numel(y_true) ~= numel(y_pred)
        metrics.accuracy  = NaN;
        metrics.precision = NaN;
        metrics.recall    = NaN;
        metrics.f1        = NaN;
        metrics.confusion_matrix = zeros(2);
        return;
    end

    labels = unique([y_true; y_pred]);

    if numel(labels) < 2
        metrics.accuracy  = mean(y_true == y_pred);
        metrics.precision = metrics.accuracy;
        metrics.recall    = metrics.accuracy;
        metrics.f1        = metrics.accuracy;
        metrics.confusion_matrix = zeros(2);
        return;
    end

    pos = labels(end);

    tp = sum(y_true == pos & y_pred == pos);
    tn = sum(y_true ~= pos & y_pred ~= pos);
    fp = sum(y_true ~= pos & y_pred == pos);
    fn = sum(y_true == pos & y_pred ~= pos);

    metrics.accuracy  = (tp + tn) / numel(y_true);
    metrics.precision = tp / max(tp + fp, eps);
    metrics.recall    = tp / max(tp + fn, eps);
    metrics.f1        = 2 * metrics.precision * metrics.recall / ...
                        max(metrics.precision + metrics.recall, eps);
    metrics.confusion_matrix = [tn, fp; fn, tp];
end
