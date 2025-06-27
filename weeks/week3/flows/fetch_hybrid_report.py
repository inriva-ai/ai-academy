from metaflow import Flow

run = Flow('HybridEvaluationFlow').latest_run
print(run.data.formatted_report)