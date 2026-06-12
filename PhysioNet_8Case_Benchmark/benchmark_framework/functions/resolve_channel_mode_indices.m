function [selected_idx, selected_labels] = resolve_channel_mode_indices(channel_mode, channel_labels)
%RESOLVE_CHANNEL_MODE_INDICES Map cfg.channel_mode to fixed channel indices.
% This fixed mask is applied before feature extraction so CSP, covariance,
% and temporal descriptors all operate on the same channel subset.

if nargin < 1 || strlength(string(channel_mode)) == 0
    channel_mode = "all";
end
if nargin < 2 || isempty(channel_labels)
    selected_idx = [];
    selected_labels = strings(0, 1);
    return;
end

channel_mode = lower(string(channel_mode));
channel_labels = string(channel_labels(:));
normalized = normalize_channel_labels(channel_labels);

switch channel_mode
    case "all"
        selected_idx = 1:numel(channel_labels);
    case "motor_basic"
        target_labels = ["C3", "Cz", "C4"];
        selected_idx = labels_to_indices(target_labels, normalized, channel_mode);
    case "motor_core"
        target_labels = ["C3", "Cz", "C4"];
        selected_idx = labels_to_indices(target_labels, normalized, channel_mode);
    case "sensorimotor"
        target_labels = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6", ...
            "C5","C3","C1","Cz","C2","C4","C6", ...
            "CP5","CP3","CP1","CPz","CP2","CP4","CP6"];
        selected_idx = labels_to_indices(target_labels, normalized, channel_mode);
    case "motor_extended"
        target_labels = ["FC3", "FC1", "FCz", "FC2", "FC4", ...
            "C3", "C1", "Cz", "C2", "C4", ...
            "CP3", "CP1", "CPz", "CP2", "CP4"];
        selected_idx = labels_to_indices(target_labels, normalized, channel_mode);
    otherwise
        error('Unknown channel_mode: %s', channel_mode);
end

selected_labels = channel_labels(selected_idx);
end

function idx = labels_to_indices(target_labels, normalized_labels, channel_mode)
idx = zeros(1, numel(target_labels));
for k = 1:numel(target_labels)
    hit = find(normalized_labels == normalize_channel_labels(target_labels(k)), 1);
    if isempty(hit)
        error('Channel %s is not available for channel_mode %s.', target_labels(k), channel_mode);
    end
    idx(k) = hit;
end
end

function normalized = normalize_channel_labels(labels)
normalized = lower(regexprep(string(labels), '[^A-Za-z0-9]', ''));
end
