function log_message(log_file, message)
fid = fopen(log_file, 'a');
if fid == -1
    return;
end
fprintf(fid, '[%s] %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), message);
fclose(fid);
end
