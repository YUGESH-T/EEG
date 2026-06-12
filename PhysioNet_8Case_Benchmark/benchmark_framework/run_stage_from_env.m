stage_name = getenv("BENCH_STAGE_NAME");
if strlength(stage_name) == 0
    stage_name = "smoke";
end
run_benchmark_stage(stage_name);
