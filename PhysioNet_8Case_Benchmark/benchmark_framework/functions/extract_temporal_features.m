function features = extract_temporal_features(X, cfg)
params = resolve_temporal_params(cfg);
n_trials = size(X, 3);
n_channels = size(X, 1);
features = zeros(n_trials, n_channels * 4);

for trial_idx = 1:n_trials
    epoch = squeeze(X(:, :, trial_idx));
    dx = diff(epoch, 1, 2);
    ddx = diff(dx, 1, 2);

    activity = var(epoch, 0, 2)' + eps;
    mobility = sqrt((var(dx, 0, 2)' + eps) ./ activity);
    complexity = sqrt((var(ddx, 0, 2)' + eps) ./ (var(dx, 0, 2)' + eps)) ./ (mobility + eps);

    sample_entropy = zeros(1, n_channels);
    for channel_idx = 1:n_channels
        sample_entropy(channel_idx) = compute_sample_entropy( ...
            epoch(channel_idx, :), ...
            params.m, ...
            params.r_factor, ...
            params.stride, ...
            params.max_points);
    end

    features(trial_idx, :) = [activity, mobility, complexity, sample_entropy];
end
end

function params = resolve_temporal_params(cfg)
params = struct('m', 2, 'r_factor', 0.2, 'stride', 8, 'max_points', 80);
if isfield(cfg, 'temporal')
    if isfield(cfg.temporal, 'sample_entropy_m') && ~isempty(cfg.temporal.sample_entropy_m)
        params.m = cfg.temporal.sample_entropy_m;
    end
    if isfield(cfg.temporal, 'sample_entropy_r') && ~isempty(cfg.temporal.sample_entropy_r)
        params.r_factor = cfg.temporal.sample_entropy_r;
    end
    if isfield(cfg.temporal, 'sample_entropy_stride') && ~isempty(cfg.temporal.sample_entropy_stride)
        params.stride = cfg.temporal.sample_entropy_stride;
    end
    if isfield(cfg.temporal, 'sample_entropy_max_points') && ~isempty(cfg.temporal.sample_entropy_max_points)
        params.max_points = cfg.temporal.sample_entropy_max_points;
    end
end
end

function value = compute_sample_entropy(signal, m, r_factor, stride, max_points)
signal = signal(:)';
if stride > 1
    signal = signal(1:stride:end);
end
if numel(signal) > max_points
    sample_idx = round(linspace(1, numel(signal), max_points));
    signal = signal(sample_idx);
end
signal = signal - mean(signal);
tolerance = r_factor * std(signal);
if tolerance <= eps || numel(signal) <= m + 2
    value = 0;
    return;
end
count_m = count_template_matches(signal, m, tolerance);
count_m1 = count_template_matches(signal, m + 1, tolerance);
value = -log((count_m1 + eps) / (count_m + eps));
end

function count = count_template_matches(signal, m, tolerance)
n_templates = numel(signal) - m + 1;
count = 0;
for i = 1:n_templates - 1
    template_i = signal(i:i + m - 1);
    for j = i + 1:n_templates
        template_j = signal(j:j + m - 1);
        if max(abs(template_i - template_j)) <= tolerance
            count = count + 1;
        end
    end
end
end
