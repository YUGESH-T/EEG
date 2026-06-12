function data = timetable_to_matrix(tt)
n_channels = width(tt);
n_records = height(tt);
samples_per_record = numel(tt{1, 1}{1});
data = zeros(n_channels, n_records * samples_per_record);

for ch = 1:n_channels
    signal = zeros(n_records * samples_per_record, 1);
    cursor = 1;
    for rec = 1:n_records
        chunk = tt{rec, ch}{1};
        next_cursor = cursor + numel(chunk) - 1;
        signal(cursor:next_cursor) = chunk;
        cursor = next_cursor + 1;
    end
    data(ch, :) = signal.';
end
end
