function subject_data = load_physionet_subject(subject_num, cfg)
subject_code = sprintf('S%03d', subject_num);
subject_dir = fullfile(cfg.cache_dir, subject_code);
ensure_dir(subject_dir);

fs = 160;
channel_labels = physionet_channel_labels();
channel_mode = "all";
if isfield(cfg, 'channel_mode') && strlength(string(cfg.channel_mode)) > 0
    channel_mode = string(cfg.channel_mode);
end
selected_idx = resolve_channel_mode_indices(channel_mode, channel_labels);
selected_labels = channel_labels(selected_idx);
if isfield(cfg, 'mi_window_sec') && numel(cfg.mi_window_sec) == 2
    mi_window_sec = cfg.mi_window_sec;
else
    mi_window_sec = [0 cfg.trial_duration_sec];
end
samples_per_trial = round((mi_window_sec(2) - mi_window_sec(1)) * fs);
left_trials = {};
right_trials = {};

for run_id = cfg.run_ids
    file_name = sprintf('%sR%02d.edf', subject_code, run_id);
    local_path = fullfile(subject_dir, file_name);
    url = sprintf('https://physionet.org/files/eegmmidb/1.0.0/%s/%s', subject_code, file_name);

    if ~exist(local_path, 'file')
        download_with_retry(url, local_path, 5, 60);
    end

    try
        info = edfinfo(local_path);
        tt = edfread(local_path);
    catch
        if exist(local_path, 'file')
            delete(local_path);
        end
        download_with_retry(url, local_path, 5, 60);
        info = edfinfo(local_path);
        tt = edfread(local_path);
    end

    raw = timetable_to_matrix(tt);
    raw = raw(selected_idx, :);
    anns = info.Annotations;

    for i = 1:height(anns)
        label = string(anns.Annotations(i));
        if label ~= "T1" && label ~= "T2"
            continue;
        end
        onset_sec = seconds(anns.Properties.RowTimes(i));
        % Epoching stage: shift from the annotation onset into the requested
        % motor-imagery window while leaving the legacy [0 4] behavior intact.
        start_idx = floor((onset_sec + mi_window_sec(1)) * fs) + 1;
        stop_idx = start_idx + samples_per_trial - 1;
        if stop_idx > size(raw, 2)
            continue;
        end
        trial = raw(:, start_idx:stop_idx);
        if label == "T1"
            left_trials{end + 1} = trial; %#ok<AGROW>
        else
            right_trials{end + 1} = trial; %#ok<AGROW>
        end
    end
end

n_trials = min(numel(left_trials), numel(right_trials));
if n_trials < cfg.min_balanced_trials
    error('Insufficient balanced trials for S%03d: left=%d right=%d', subject_num, numel(left_trials), numel(right_trials));
end

left_trials = left_trials(1:n_trials);
right_trials = right_trials(1:n_trials);

X_left = zeros(size(left_trials{1}, 1), size(left_trials{1}, 2), n_trials);
X_right = zeros(size(right_trials{1}, 1), size(right_trials{1}, 2), n_trials);
for i = 1:n_trials
    X_left(:, :, i) = left_trials{i};
    X_right(:, :, i) = right_trials{i};
end

subject_data = struct();
subject_data.subject_num = subject_num;
subject_data.subject_name = sprintf('s%03d', subject_num);
subject_data.fs = fs;
subject_data.X_left = X_left;
subject_data.X_right = X_right;
subject_data.n_trials = n_trials;
subject_data.channel_labels = selected_labels;
subject_data.channel_mode = string(channel_mode);
end

function labels = physionet_channel_labels()
labels = [ ...
    "FC5","FC3","FC1","FCz","FC2","FC4","FC6", ...
    "C5","C3","C1","Cz","C2","C4","C6", ...
    "CP5","CP3","CP1","CPz","CP2","CP4","CP6", ...
    "FP1","FPz","FP2","AF7","AF3","AFz","AF4","AF8", ...
    "F7","F5","F3","F1","Fz","F2","F4","F6","F8", ...
    "FT7","FT8","T7","T8","T9","T10","TP7","TP8", ...
    "P7","P5","P3","P1","Pz","P2","P4","P6","P8", ...
    "PO7","PO3","POz","PO4","PO8","O1","Oz","O2", ...
    "Iz"];
end
