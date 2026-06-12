function [metrics, conf_mat] = compute_binary_metrics(y_true, y_pred, positive_label)
labels = [1 2];
conf_mat = confusionmat(y_true, y_pred, 'Order', labels);

pos_idx = find(labels == positive_label, 1);
tp = conf_mat(pos_idx, pos_idx);
fp = sum(conf_mat(:, pos_idx)) - tp;
fn = sum(conf_mat(pos_idx, :)) - tp;

accuracy = sum(diag(conf_mat)) / max(1, sum(conf_mat(:)));
precision = tp / max(1, tp + fp);
recall = tp / max(1, tp + fn);
f1 = (2 * precision * recall) / max(eps, precision + recall);

metrics = struct();
metrics.accuracy = accuracy;
metrics.precision = precision;
metrics.recall = recall;
metrics.f1 = f1;
end
