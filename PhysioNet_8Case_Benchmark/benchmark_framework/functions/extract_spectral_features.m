function features = extract_spectral_features(X, cfg)
bands = resolve_spectral_bands(cfg);
fs = cfg.fs;
n_trials = size(X, 3);
n_channels = size(X, 1);
n_bands = size(bands, 1);
features = zeros(n_trials, n_channels * n_bands * 2);

write_idx = 1;
for band_idx = 1:n_bands
    [b, a] = butter(4, bands(band_idx, :) / (fs / 2), 'bandpass');
    band_features = zeros(n_trials, n_channels * 2);
    for trial_idx = 1:n_trials
        epoch = squeeze(X(:, :, trial_idx));
        filtered = filtfilt(b, a, epoch')';
        channel_var = var(filtered, 0, 2) + eps;
        de_block = 0.5 * log(2 * pi * exp(1) * channel_var);
        bp_block = log(mean(filtered .^ 2, 2) + eps);
        band_features(trial_idx, :) = [de_block; bp_block].';
    end
    cols = write_idx:(write_idx + size(band_features, 2) - 1);
    features(:, cols) = band_features;
    write_idx = cols(end) + 1;
end
end

function bands = resolve_spectral_bands(cfg)
if isfield(cfg, 'spectral') && isfield(cfg.spectral, 'bands') && ~isempty(cfg.spectral.bands)
    bands = cfg.spectral.bands;
    return;
end
bands = [4 8; 8 12; 12 16; 16 20; 20 24; 24 30];
end
