function ensure_dir(path_str)
if ~exist(path_str, 'dir')
    mkdir(path_str);
end
end
