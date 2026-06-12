function log_experiment(log_file, message)
%LOG_EXPERIMENT Append a timestamped message to the experiment log file.
%
%   log_experiment(log_file, message)
%
%   Writes both to disk (append) and to stdout for real-time monitoring.
%   Structured format: [YYYY-MM-DD HH:MM:SS] message

    timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    formatted = sprintf('[%s] %s', timestamp, message);

    fid = fopen(log_file, 'a');
    if fid ~= -1
        fprintf(fid, '%s\n', formatted);
        fclose(fid);
    end

    fprintf('%s\n', formatted);
end
