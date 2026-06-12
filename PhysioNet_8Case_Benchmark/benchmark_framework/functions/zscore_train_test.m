function [train_X_norm, test_X_norm, stats] = zscore_train_test(train_X, test_X)
mu = mean(train_X, 1);
sigma = std(train_X, 0, 1);
sigma(sigma == 0) = 1;
train_X_norm = (train_X - mu) ./ sigma;
test_X_norm = (test_X - mu) ./ sigma;
stats = struct('mu', mu, 'sigma', sigma);
end
