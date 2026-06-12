function features = extract_spectral_features_v2(Z, fs, spectral_mode)
%EXTRACT_SPECTRAL_FEATURES_V2 Sub-band spectral features from CSP signals.
%
%   features = extract_spectral_features_v2(Z, fs, spectral_mode)
%
%   Inputs:
%     Z             - [n_components × n_time × n_epochs] CSP-projected signals
%     fs            - sampling frequency (Hz)
%     spectral_mode - '3band' or '6band'
%
%   3band: [8,12; 12,20; 20,30] Hz — log band-power only (baseline match)
%     Output: [n_epochs × n_components*3]
%
%   6band: [4,8; 8,12; 12,16; 16,20; 20,24; 24,30] Hz — log BP + DE
%     Output: [n_epochs × n_components*6*2]
%
%   Spectral features are extracted from broadband CSP projections, allowing
%   sub-band decomposition across the full theta-beta range.

    if nargin < 3 || isempty(spectral_mode)
        spectral_mode = '6band';
    end

    spectral_mode = lower(string(spectral_mode));

    switch spectral_mode
        case "none"
            features = [];
            return;
        case "3band"
            bands  = [8 12; 12 20; 20 30];
            use_de = false;
        case "6band"
            bands  = [4 8; 8 12; 12 16; 16 20; 20 24; 24 30];
            use_de = true;
        otherwise
            error('extract_spectral_features_v2:unknownMode', ...
                'Unknown spectral mode: %s. Use ''none'', ''3band'' or ''6band''.', spectral_mode);
    end

    n_components = size(Z, 1);
    n_time       = size(Z, 2);
    n_epochs     = size(Z, 3);
    n_bands      = size(bands, 1);
    nyquist      = fs / 2;

    feat_per_band = 1;
    if use_de
        feat_per_band = 2;  % log-bandpower + differential entropy
    end

    n_features = n_components * n_bands * feat_per_band;
    features   = zeros(n_epochs, n_features);

    % Minimum signal length for filtfilt: 3 * filter_order
    % We use order 3-4, so minimum ~12-15 samples. Guard against short epochs.
    min_samples_for_filtfilt = 20;

    for b = 1:n_bands
        band = bands(b, :);

        % Guard: skip bands that violate Nyquist
        if band(1) >= nyquist || band(2) > nyquist
            continue;
        end

        % Design bandpass filter — lower order for narrower bands
        bw = band(2) - band(1);
        if bw < 5
            filt_order = 3;
        else
            filt_order = 4;
        end

        try
            [bf, af] = butter(filt_order, band / nyquist, 'bandpass');
        catch
            % Fallback: reduce order
            [bf, af] = butter(2, band / nyquist, 'bandpass');
        end

        for e = 1:n_epochs
            for c = 1:n_components
                signal = squeeze(Z(c, :, e));
                signal = signal(:);  % column vector

                % Apply bandpass
                if n_time >= min_samples_for_filtfilt
                    try
                        filtered = filtfilt(bf, af, signal);
                    catch
                        filtered = filter(bf, af, signal);
                    end
                else
                    filtered = filter(bf, af, signal);
                end

                % Log band-power: log(mean(x²))
                bp = mean(filtered.^2) + eps;

                feat_base = ((c - 1) * n_bands + (b - 1)) * feat_per_band;
                features(e, feat_base + 1) = log(bp);

                if use_de
                    % Differential entropy: 0.5 * log(2πe * var(x))
                    v = var(filtered) + eps;
                    features(e, feat_base + 2) = 0.5 * log(2 * pi * exp(1) * v);
                end
            end
        end
    end

    % NaN/Inf guard
    features(~isfinite(features)) = 0;
end
