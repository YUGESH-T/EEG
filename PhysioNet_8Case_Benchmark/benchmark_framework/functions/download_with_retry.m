function download_with_retry(url, local_path, max_attempts, timeout_sec)
last_error = [];
for attempt = 1:max_attempts
    try
        opts = weboptions('Timeout', timeout_sec);
        websave(local_path, url, opts);
        return;
    catch ME
        last_error = ME;
        if attempt < max_attempts
            pause(2 * attempt);
        end
    end
end
rethrow(last_error);
end
