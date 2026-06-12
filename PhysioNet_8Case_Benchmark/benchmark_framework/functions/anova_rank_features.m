function selected_idx = anova_rank_features(X, Y, top_k)
D = size(X, 2);
F = zeros(1, D);
idx1 = (Y == 1);
idx2 = (Y == 2);
N1 = sum(idx1);
N2 = sum(idx2);
for d = 1:D
    col = X(:, d);
    mu = mean(col);
    mu1 = mean(col(idx1));
    mu2 = mean(col(idx2));
    var1 = var(col(idx1));
    var2 = var(col(idx2));
    between = N1 * (mu1 - mu)^2 + N2 * (mu2 - mu)^2;
    within = (N1 - 1) * var1 + (N2 - 1) * var2;
    if within > 0
        F(d) = between / within;
    else
        F(d) = 0;
    end
end
[~, order] = sort(F, 'descend');
selected_idx = order(1:min(top_k, D));
end
