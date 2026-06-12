function mean_cov = riemannian_mean_spd(cov_matrices)
num_matrices = size(cov_matrices, 3);
mean_cov = mean(cov_matrices, 3);
for iter = 1:5
    P_half = sqrtm(mean_cov);
    P_half_inv = inv(P_half);
    sum_log = zeros(size(mean_cov));
    for i = 1:num_matrices
        tmp = P_half_inv * cov_matrices(:, :, i) * P_half_inv;
        sum_log = sum_log + logm(tmp);
    end
    mean_cov = P_half * expm(sum_log / num_matrices) * P_half;
    mean_cov = (mean_cov + mean_cov') / 2;
end
end
