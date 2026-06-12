function [W, eigvals] = compute_csp_filters(class1, class2)
n1 = size(class1, 3);
n2 = size(class2, 3);
cov1 = zeros(size(class1, 1));
cov2 = zeros(size(class2, 1));
for i = 1:n1
    trial = squeeze(class1(:, :, i));
    cov1 = cov1 + (trial * trial') / trace(trial * trial');
end
for i = 1:n2
    trial = squeeze(class2(:, :, i));
    cov2 = cov2 + (trial * trial') / trace(trial * trial');
end
cov1 = cov1 / n1;
cov2 = cov2 / n2;
[W_raw, D] = eig(cov1, cov1 + cov2);
[eigvals, idx] = sort(diag(D), 'descend');
W = W_raw(:, idx)';
end
