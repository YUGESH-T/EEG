function [X_left_f, X_right_f] = apply_bandpass_filter(X_left, X_right, fs, filter_type, cfg)
X_left_f = X_left;
X_right_f = X_right;

switch lower(filter_type)
    case 'common'
        bpFilt = designfilt('bandpassiir', 'FilterOrder', cfg.common_filter_order, ...
            'HalfPowerFrequency1', cfg.band(1), 'HalfPowerFrequency2', cfg.band(2), ...
            'SampleRate', fs);
        for i = 1:size(X_left, 3)
            for ch = 1:size(X_left, 1)
                X_left_f(ch, :, i) = filtfilt(bpFilt, double(X_left(ch, :, i)));
                X_right_f(ch, :, i) = filtfilt(bpFilt, double(X_right(ch, :, i)));
            end
        end
    case 'butterworth'
        [b, a] = butter(cfg.butterworth_order, cfg.band / (fs / 2), 'bandpass');
        for i = 1:size(X_left, 3)
            for ch = 1:size(X_left, 1)
                X_left_f(ch, :, i) = filtfilt(b, a, double(X_left(ch, :, i)));
                X_right_f(ch, :, i) = filtfilt(b, a, double(X_right(ch, :, i)));
            end
        end
    otherwise
        error('Unknown filter_type: %s', filter_type);
end
end
