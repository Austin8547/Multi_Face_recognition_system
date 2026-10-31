from src.evaluation.latency.settings import run_benchmark

def main():
    run_benchmark(
        model_name="buffalo_s",
        display_title="InsightFace (buffalo_s) — latency benchmark (q to stop)",
        csv_prefix="insightface_buffalo_s_latency"
    )

if __name__ == "__main__":
    main()
    
    
    