from src.evaluation.latency.settings import run_benchmark

def main():
    run_benchmark(
        model_name="buffalo_m",
        display_title="InsightFace (buffalo_m) — latency benchmark (q to stop)",
        csv_prefix="insightface_buffalo_m_latency"
    )

if __name__ == "__main__":
    main()
    
    
    