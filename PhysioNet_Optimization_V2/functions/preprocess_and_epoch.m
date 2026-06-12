function [epochs, labels] = preprocess_and_epoch(raw_eeg, events, mi_window, cfg)
%PREPROCESS_AND_EPOCH Apply notch → bandpass → CAR → epoching pipeline.
%
%   [epochs, labels] = preprocess_and_epoch(raw_eeg, events, mi_window, cfg)
%
%   Inputs:
%     raw_eeg   - [n_channels × n_total_samples] raw continuous EEG
%     events    - [n_events × 2] columns: [onset_sample, class_label]
%     mi_window - [tmin, tmax] seconds relative to event onset
%     cfg       - struct with fields:
%                   .fs                    (sampling frequency, 160)
%                   .preprocessing_bandpass ([4 32])
%                   .butterworth_order     (4)
%
%   Outputs:
%     epochs - [n_channels × epoch_len × n_valid_epochs]
%     labels - [n_valid_epochs × 1]

    fs   = cfg.fs;
    data = double(raw_eeg);

    % ----- Notch filter (50 Hz powerline) --------------------------------
    notch_freq = 50;
    if notch_freq > 0 && notch_freq < fs / 2
        q_factor  = 35;
        bandwidth = notch_freq / q_factor;
        lo = max(notch_freq - bandwidth / 2, 0.1);
        hi = min(notch_freq + bandwidth / 2, fs / 2 - 0.1);
        [b, a] = butter(2, [lo hi] / (fs / 2), 'stop');
        data = filtfilt(b, a, data')';
    end

    % ----- Butterworth bandpass ------------------------------------------
    bp    = cfg.preprocessing_bandpass;
    order = cfg.butterworth_order;
    [b, a] = butter(order, bp / (fs / 2), 'bandpass');
    data = filtfilt(b, a, data')';

    % ----- Common Average Reference (CAR) --------------------------------
    data = data - mean(data, 1);

    % ----- Epoching with MI window ---------------------------------------
    tmin         = mi_window(1);
    tmax         = mi_window(2);
    tmin_samples = round(tmin * fs);
    tmax_samples = round(tmax * fs);
    epoch_len    = tmax_samples - tmin_samples;

    n_events   = size(events, 1);
    n_channels = size(data, 1);

    epochs = zeros(n_channels, epoch_len, n_events);
    labels = zeros(n_events, 1);
    keep   = false(n_events, 1);

    for i = 1:n_events
        start_idx = events(i, 1) + tmin_samples;
        stop_idx  = start_idx + epoch_len - 1;

        if start_idx >= 1 && stop_idx <= size(data, 2)
            epoch = data(:, start_idx:stop_idx);
            % Per-epoch baseline removal (zero-mean each channel)
            epoch = epoch - mean(epoch, 2);
            epochs(:, :, i) = epoch;
            labels(i)       = events(i, 2);
            keep(i)         = true;
        end
    end

    epochs = epochs(:, :, keep);
    labels = labels(keep);

    if isempty(labels)
        error('preprocess_and_epoch:noEpochs', ...
            'No valid epochs extracted for MI window [%.2f, %.2f] s.', tmin, tmax);
    end
end
