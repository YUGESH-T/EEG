function benchmark_app()
%BENCHMARK_APP  Interactive GUI for the PhysioNet 8-Case Benchmark.
%   Type  benchmark_app  in the MATLAB Command Window to launch.

    % --- Resolve paths --------------------------------------------------------
    framework_dir = fileparts(mfilename('fullpath'));
    functions_dir = fullfile(framework_dir, 'functions');
    workspace_dir = fileparts(framework_dir);
    addpath(framework_dir);
    addpath(functions_dir);

    % --- App state ------------------------------------------------------------
    state.running        = false;
    state.finishing      = false;
    state.worker         = [];
    state.log_timer      = [];
    state.log_lines_read = 0;
    state.stage_log_path = '';

    % --- Case matrix (must match build_case_configs.m) ------------------------
    case_labels = {
        'Case 01 :  CSP + SVM + Common'
        'Case 02 :  Riemannian + SVM + Common'
        'Case 03 :  CSP + RF + Common'
        'Case 04 :  Riemannian + RF + Common'
        'Case 05 :  CSP + SVM + Butterworth'
        'Case 06 :  Riemannian + SVM + Butterworth'
        'Case 07 :  CSP + RF + Butterworth'
        'Case 08 :  Riemannian + RF + Butterworth'
    };

    % --- Dark-theme palette ---------------------------------------------------
    C_BG     = [0.11 0.11 0.14];
    C_PANEL  = [0.16 0.16 0.20];
    C_TEXT   = [0.90 0.90 0.94];
    C_DIM    = [0.52 0.52 0.58];
    C_GREEN  = [0.16 0.65 0.42];
    C_RED    = [0.82 0.24 0.24];
    C_LOGBG  = [0.09 0.09 0.11];

    % =====================================================================
    %  BUILD THE INTERFACE
    % =====================================================================
    fig = uifigure( ...
        'Name',            'PhysioNet 8-Case Benchmark Runner', ...
        'Position',        [120 60 960 740], ...
        'Color',           C_BG, ...
        'Resize',          'off', ...
        'CloseRequestFcn', @onClose);

    % --- Title bar ---
    uilabel(fig, ...
        'Text',     'PhysioNet 8-Case Benchmark', ...
        'Position', [30 690 500 38], ...
        'FontSize', 22, 'FontWeight', 'bold', 'FontColor', C_TEXT);
    uilabel(fig, ...
        'Text',     'EEG Motor-Imagery Classification  |  Pipeline Runner', ...
        'Position', [30 670 500 20], ...
        'FontSize', 11, 'FontColor', C_DIM);

    % =================================================================
    %  LEFT PANEL  —  Configuration
    % =================================================================
    pnl_cfg = uipanel(fig, 'Title', '', ...
        'Position', [20 280 440 380], ...
        'BackgroundColor', C_PANEL, 'BorderType', 'none');

    % Stage selector
    uilabel(pnl_cfg, 'Text', 'STAGE', ...
        'Position', [15 350 100 16], ...
        'FontSize', 10, 'FontWeight', 'bold', 'FontColor', C_DIM);

    dd_stage = uidropdown(pnl_cfg, ...
        'Items',     {'smoke  (3 subjects)', ...
                      'pilot  (10 subjects)', ...
                      'full   (109 subjects)'}, ...
        'ItemsData', {'smoke','pilot','full'}, ...
        'Value',     'smoke', ...
        'Position',  [15 320 210 28], ...
        'ValueChangedFcn', @onStageChanged);

    % Case checkboxes
    uilabel(pnl_cfg, 'Text', 'CASES', ...
        'Position', [15 286 100 16], ...
        'FontSize', 10, 'FontWeight', 'bold', 'FontColor', C_DIM);

    cb = gobjects(8, 1);
    for i = 1:8
        cb(i) = uicheckbox(pnl_cfg, ...
            'Text',     case_labels{i}, ...
            'Value',    true, ...
            'Position', [22  286 - 28*i  410  22], ...
            'FontColor', C_TEXT);
    end

    % Select All / None
    uibutton(pnl_cfg, 'Text', 'All', ...
        'Position', [22 22 60 24], ...
        'ButtonPushedFcn', @(~,~) setCases(true));
    uibutton(pnl_cfg, 'Text', 'None', ...
        'Position', [90 22 60 24], ...
        'ButtonPushedFcn', @(~,~) setCases(false));

    % =================================================================
    %  RIGHT PANEL  —  Actions & Status
    % =================================================================
    pnl_act = uipanel(fig, 'Title', '', ...
        'Position', [480 280 460 380], ...
        'BackgroundColor', C_PANEL, 'BorderType', 'none');

    % Subject override
    uilabel(pnl_act, 'Text', 'SUBJECT OVERRIDE  (optional)', ...
        'Position', [15 350 300 16], ...
        'FontSize', 10, 'FontWeight', 'bold', 'FontColor', C_DIM);
    uilabel(pnl_act, 'Text', 'Comma-separated IDs.  Leave empty for stage default.', ...
        'Position', [15 330 420 16], ...
        'FontSize', 10, 'FontColor', C_DIM);

    ef_subj = uieditfield(pnl_act, 'text', ...
        'Value',       '', ...
        'Placeholder', 'Default: subjects 1-3', ...
        'Position',    [15 300 260 28]);

    % Status display
    uilabel(pnl_act, 'Text', 'STATUS', ...
        'Position', [15 258 100 16], ...
        'FontSize', 10, 'FontWeight', 'bold', 'FontColor', C_DIM);

    lbl_status = uilabel(pnl_act, ...
        'Text',       'Idle', ...
        'Position',   [15 230 420 24], ...
        'FontSize',   15, 'FontWeight', 'bold', 'FontColor', C_DIM);

    lbl_progress = uilabel(pnl_act, ...
        'Text',     '', ...
        'Position', [15 206 420 20], ...
        'FontSize', 11, 'FontColor', C_DIM);

    % Run / Stop buttons
    btn_run = uibutton(pnl_act, ...
        'Text',            'Run Benchmark', ...
        'Position',        [15 148 180 46], ...
        'FontSize',        15, 'FontWeight', 'bold', ...
        'BackgroundColor', C_GREEN, 'FontColor', [1 1 1], ...
        'ButtonPushedFcn', @onRun);

    btn_stop = uibutton(pnl_act, ...
        'Text',            'Stop', ...
        'Position',        [210 148 110 46], ...
        'FontSize',        15, 'FontWeight', 'bold', ...
        'BackgroundColor', C_RED, 'FontColor', [1 1 1], ...
        'Enable',          'off', ...
        'ButtonPushedFcn', @onStop);

    % Open results folder
    uibutton(pnl_act, ...
        'Text',            'Open Results Folder', ...
        'Position',        [15 92 180 30], ...
        'ButtonPushedFcn', @onOpenResults);

    % Elapsed time
    lbl_elapsed = uilabel(pnl_act, ...
        'Text',     '', ...
        'Position', [15 60 300 20], ...
        'FontSize', 11, 'FontColor', C_DIM);

    % =================================================================
    %  BOTTOM PANEL  —  Log
    % =================================================================
    pnl_log = uipanel(fig, 'Title', '', ...
        'Position', [20 10 920 262], ...
        'BackgroundColor', C_PANEL, 'BorderType', 'none');

    uilabel(pnl_log, 'Text', 'LOG OUTPUT', ...
        'Position', [15 236 100 16], ...
        'FontSize', 10, 'FontWeight', 'bold', 'FontColor', C_DIM);

    uibutton(pnl_log, 'Text', 'Clear', ...
        'Position', [852 234 55 22], ...
        'ButtonPushedFcn', @(~,~) set(ta_log,'Value',{''}));

    ta_log = uitextarea(pnl_log, ...
        'Value',           {''}, ...
        'Editable',        'off', ...
        'Position',        [15 8 892 224], ...
        'FontName',        'Consolas', ...
        'FontSize',        11, ...
        'BackgroundColor', C_LOGBG, ...
        'FontColor',       C_TEXT);

    % track start time
    state.start_tic = [];

    % =====================================================================
    %  CALLBACK FUNCTIONS
    % =====================================================================

    function onStageChanged(~, ~)
        switch dd_stage.Value
            case 'smoke', ef_subj.Placeholder = 'Default: subjects 1-3';
            case 'pilot', ef_subj.Placeholder = 'Default: subjects 1-10';
            case 'full',  ef_subj.Placeholder = 'Default: subjects 1-109';
        end
    end

    function setCases(val)
        for k = 1:8
            cb(k).Value = val;
        end
    end

    function appendLog(msg)
        cur = ta_log.Value;
        if numel(cur) == 1 && strlength(string(cur{1})) == 0
            ta_log.Value = {msg};
        else
            ta_log.Value = [cur; {msg}];
        end
        try scroll(ta_log, 'bottom'); catch; end %#ok
    end

    function lockUI(on)
        if on, v = 'off'; else, v = 'on'; end
        btn_run.Enable  = v;
        dd_stage.Enable = v;
        ef_subj.Enable  = v;
        for k = 1:8; cb(k).Enable = v; end
        if on
            btn_stop.Enable = 'on';
        else
            btn_stop.Enable = 'off';
        end
    end

    % -----------------------------------------------------------------
    function onRun(~, ~)
        if state.running; return; end

        % Gather selected cases
        sel = [];
        for k = 1:8
            if cb(k).Value
                sel(end+1) = k; %#ok<AGROW>
            end
        end
        if isempty(sel)
            uialert(fig, 'Select at least one case.', 'No Cases');
            return;
        end

        stage    = dd_stage.Value;
        subj_str = strtrim(ef_subj.Value);

        % Environment overrides
        if numel(sel) < 8
            setenv('BENCH_CASE_LIST', strjoin(string(sel), ','));
        else
            setenv('BENCH_CASE_LIST', '');
        end
        if strlength(string(subj_str)) > 0
            setenv('BENCH_SUBJECT_LIST', subj_str);
        else
            setenv('BENCH_SUBJECT_LIST', '');
        end

        % Log path
        stage_root = fullfile(workspace_dir, 'benchmark_results', ...
            ['stage_' stage]);
        state.stage_log_path = fullfile(stage_root, 'stage_log.txt');
        state.log_lines_read = 0;

        % UI feedback
        state.running   = true;
        state.finishing = false;
        state.start_tic = tic;
        lockUI(true);
        lbl_status.Text      = 'Running...';
        lbl_status.FontColor = C_GREEN;
        lbl_progress.Text    = sprintf('Stage: %s  |  Cases: %d  |  Started %s', ...
            stage, numel(sel), datestr(now, 'HH:MM:SS')); %#ok<TNOW1,DATST>
        lbl_elapsed.Text     = 'Elapsed: 0 min';
        appendLog(sprintf('[%s]  Launching stage "%s" with %d case(s)...', ...
            datestr(now,'HH:MM:SS'), stage, numel(sel))); %#ok<TNOW1,DATST>

        % Start log-polling timer (every 2 s) for live UI updates
        state.log_timer = timer( ...
            'ExecutionMode', 'fixedSpacing', ...
            'Period',        2, ...
            'TimerFcn',      @onTick, ...
            'ErrorFcn',      @(~,~) []);
        start(state.log_timer);

        % Run synchronously in main thread.
        % drawnow calls inside run_benchmark_stage keep the UI and
        % log-polling timer responsive between subjects.
        appendLog('  -> Running in main thread.  Log updates between subjects.');
        drawnow;
        try
            run_benchmark_stage(stage);
            onFinished('');
        catch ME2
            onFinished(ME2.message);
        end
    end

    % -----------------------------------------------------------------
    function onTick(~, ~)
        if ~isvalid(fig); return; end
        try
            % Elapsed time
            if ~isempty(state.start_tic)
                el = toc(state.start_tic);
                mins = floor(el / 60);
                secs = floor(mod(el, 60));
                lbl_elapsed.Text = sprintf('Elapsed: %d min %02d sec', mins, secs);
            end

            % Poll log file for new lines
            if exist(state.stage_log_path, 'file')
                lines = readlines(state.stage_log_path);
                n = numel(lines);
                if n > state.log_lines_read
                    for k = (state.log_lines_read + 1) : n
                        txt = char(lines(k));
                        if ~isempty(strtrim(txt))
                            appendLog(txt);
                        end
                    end
                    state.log_lines_read = n;
                end
            end
        catch
            % Swallow timer errors for robustness
        end
    end

    % -----------------------------------------------------------------
    function onFinished(err_msg)
        if state.finishing
            return;
        end
        state.finishing = true;
        state.worker = [];
        onTick([], []);   % final flush without re-entering completion handling
        killTimer();
        clearEnv();
        state.running = false;
        lockUI(false);

        if isempty(err_msg) || strlength(string(err_msg)) == 0
            lbl_status.Text      = 'Completed';
            lbl_status.FontColor = C_GREEN;
            appendLog(sprintf('[%s]  Benchmark completed successfully.', ...
                datestr(now,'HH:MM:SS'))); %#ok<TNOW1,DATST>
            showSummary();
        else
            lbl_status.Text      = 'Error';
            lbl_status.FontColor = C_RED;
            appendLog(sprintf('[%s]  ERROR: %s', ...
                datestr(now,'HH:MM:SS'), err_msg)); %#ok<TNOW1,DATST>
        end

        if ~isempty(state.start_tic)
            el = toc(state.start_tic);
            lbl_progress.Text = sprintf('Total time: %d min %02d sec', ...
                floor(el/60), floor(mod(el,60)));
        end
        state.finishing = false;
    end

    % -----------------------------------------------------------------
    function showSummary()
        csv_path = fullfile(fileparts(state.stage_log_path), ...
            'artifacts', 'stage_summary.csv');
        if ~exist(csv_path, 'file'); return; end
        try
            tbl = readtable(csv_path, 'VariableNamingRule', 'preserve');
            appendLog(' ');
            appendLog('=== STAGE SUMMARY ===');
            for k = 1:height(tbl)
                appendLog(sprintf('  %-38s  Acc %.3f +/- %.3f   F1 %.3f   (N=%d)', ...
                    char(tbl.CaseId(k)), tbl.MeanAccuracy(k), ...
                    tbl.StdAccuracy(k), tbl.MeanF1(k), tbl.SubjectCount(k)));
            end
            appendLog('=====================');
        catch
        end
    end

    % -----------------------------------------------------------------
    function onStop(~, ~)
        if ~state.running; return; end
        try
            if ~isempty(state.worker)
                cancel(state.worker);
            end
        catch
        end
        state.worker = [];
        state.finishing = false;
        killTimer();
        clearEnv();
        state.running = false;
        lockUI(false);
        lbl_status.Text      = 'Stopped';
        lbl_status.FontColor = C_RED;
        appendLog(sprintf('[%s]  Stopped by user.', datestr(now,'HH:MM:SS'))); %#ok<TNOW1,DATST>
    end

    % -----------------------------------------------------------------
    function onOpenResults(~, ~)
        d = fullfile(workspace_dir, 'benchmark_results', ...
            ['stage_' dd_stage.Value]);
        if exist(d, 'dir')
            winopen(d);
        else
            uialert(fig, ...
                sprintf('No results folder for stage "%s" yet.', dd_stage.Value), ...
                'Not Found');
        end
    end

    % -----------------------------------------------------------------
    function onClose(~, ~)
        if state.running
            try cancel(state.worker); catch; end
        end
        state.worker = [];
        state.finishing = false;
        killTimer();
        clearEnv();
        delete(fig);
    end

    function killTimer()
        if ~isempty(state.log_timer)
            try stop(state.log_timer); catch; end
            try delete(state.log_timer); catch; end
            state.log_timer = [];
        end
    end

    function clearEnv()
        setenv('BENCH_CASE_LIST',    '');
        setenv('BENCH_SUBJECT_LIST', '');
    end
end
