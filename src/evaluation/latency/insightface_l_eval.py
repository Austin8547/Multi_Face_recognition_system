from src.evaluation.latency.settings import run_benchmark

def main():
    run_benchmark(
        model_name="buffalo_l",
        display_title="InsightFace (buffalo_l) — latency benchmark (q to stop)",
        csv_prefix="insightface_video_latency"
    )

if __name__ == "__main__":
    main()