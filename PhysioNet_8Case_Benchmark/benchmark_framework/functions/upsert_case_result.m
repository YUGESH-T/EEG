function upsert_case_result(csv_path, row_tbl)
if exist(csv_path, 'file')
    opts = detectImportOptions(csv_path);
    opts.VariableNamingRule = 'preserve';
    tbl = readtable(csv_path, opts);
    idx = strcmp(string(tbl.Subject), string(row_tbl.Subject));
    tbl(idx, :) = [];
    tbl = [tbl; row_tbl];
else
    tbl = row_tbl;
end
tbl = sortrows(tbl, 'Subject');
writetable(tbl, csv_path);
end
