function [raw_eeg, events, subject_info] = load_physionet_raw(subject_id, cfg)
%LOAD_PHYSIONET_RAW Load raw continuous EEG and event annotations.
%
%   [raw_eeg, events, subject_info] = load_physionet_raw(subject_id, cfg)
%
%   Reads cached EDF files for motor imagery runs (default: R04, R08, R12).
%   Returns:
%     raw_eeg      - [n_channels × n_total_samples] continuous data
%     events       - [n_events × 2] columns: [onset_sample, class_label]
%     subject_info - struct with subject_id, subject_code, fs, n_trials, n_channels
%
%   Does NOT apply any filtering, epoching, or preprocessing.

    subject_code = sprintf('S%03d', subject_id);
    subject_dir  = fullfile(cfg.cache_dir, subject_code);

    if ~exist(subject_dir, 'dir')
        error('load_physionet_raw:notFound', ...
            'Subject cache folder not found: %s', subject_dir);
    end

    fs             = cfg.fs;
    raw_runs       = {};
    events         = [];
    sample_offset  = 0;
    left_count     = 0;
    right_count    = 0;

    for run_id = cfg.run_ids
        file_name  = sprintf('%sR%02d.edf', subject_code, run_id);
        local_path = fullfile(subject_dir, file_name);

        if ~exist(local_path, 'file')
            error('load_physionet_raw:missingFile', ...
                'EDF file not found: %s', local_path);
        end

        info           = edfinfo(local_path);
        timetable_data = edfread(local_path);
        run_raw        = local_timetable_to_matrix(timetable_data);

        raw_runs{end + 1} = run_raw; %#ok<AGROW>
        annotations = info.Annotations;

        for ann_idx = 1:height(annotations)
            label = string(annotations.Annotations(ann_idx));
            if label ~= "T1" && label ~= "T2"
                continue;
            end

            onset_sec    = seconds(annotations.Properties.RowTimes(ann_idx));
            onset_sample = sample_offset + floor(onset_sec * fs) + 1;

            if label == "T2"
                class_label = 2;
                right_count = right_count + 1;
            else
                class_label = 1;
                left_count  = left_count + 1;
            end

            events = [events; onset_sample, class_label]; %#ok<AGROW>
        end

        sample_offset = sample_offset + size(run_raw, 2);
    end

    balanced = min(left_count, right_count);
    if balanced < cfg.min_balanced_trials
        error('load_physionet_raw:insufficientTrials', ...
            'Insufficient balanced trials for %s: left=%d, right=%d (min=%d)', ...
            subject_code, left_count, right_count, cfg.min_balanced_trials);
    end

    raw_eeg = cell2mat(raw_runs);
    events  = balance_events_local(events, balanced);

    subject_info.subject_id   = subject_id;
    subject_info.subject_code = subject_code;
    subject_info.fs           = fs;
    subject_info.n_trials     = size(events, 1);
    subject_info.n_channels   = size(raw_eeg, 1);
end

%% ========================================================================

function data = local_timetable_to_matrix(tt)
%LOCAL_TIMETABLE_TO_MATRIX Convert EDF timetable to [channels × samples].
    n_channels = width(tt);
    n_records  = height(tt);
    first_cell = tt{1, 1};
    spr        = numel(first_cell{1});
    data       = zeros(n_channels, n_records * spr);

    for ch = 1:n_channels
        cursor = 1;
        for rec = 1:n_records
            chunk = tt{rec, ch}{1};
            n     = numel(chunk);
            data(ch, cursor:cursor + n - 1) = chunk;
            cursor = cursor + n;
        end
    end
end

function events_out = balance_events_local(events, n_per_class)
%BALANCE_EVENTS_LOCAL Truncate to equal class counts, preserve temporal order.
    c1 = events(events(:, 2) == 1, :);
    c2 = events(events(:, 2) == 2, :);

    events_out = [c1(1:n_per_class, :); c2(1:n_per_class, :)];
    [~, order]  = sort(events_out(:, 1));
    events_out  = events_out(order, :);
end
