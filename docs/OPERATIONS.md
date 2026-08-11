# Operations

Scheduled workflows use a shared `xsmb-daily-write` concurrency group to serialize repository writes. Forecast runs are idempotent because an existing live snapshot is never overwritten. Settlement runs are idempotent because `output/pipeline_state.json` records the last settled draw; the second nightly schedule is only a fallback for source delay.
