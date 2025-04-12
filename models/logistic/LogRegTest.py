import os
import yaml
import subprocess
import logging

# python -m models.Logistic.LogRegTest


def setup_logger(log_dir: str, exp_name: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_name}_batch.txt")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )
    logging.info("========== New Batch Run Started ==========")
    return log_path


def load_completed(progress_file):
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def save_completed(progress_file, language):
    with open(progress_file, 'a') as f:
        f.write(f"{language}\n")


def run_for_all_languages():
    config_path = "conifg/logreg.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_args = config["dataset"]
    logreg_cfg = config["logreg_params"]
    exp_name = logreg_cfg.get("exp_name", "logreg_batch")
    backbone = dataset_args.get("backbone", "unknown")

    # Load language mapping
    with open('./dataset/language_encode.json') as f:
        lang_map = yaml.safe_load(f)

    languages = list(lang_map.keys())

    # Setup logging and progress tracking
    log_path = setup_logger("logs", exp_name)
    print(f"[INFO] Batch logging to: {log_path}")

    os.makedirs("models/Logistic/progress", exist_ok=True)
    progress_file = os.path.join("models", "Logistic", "progress", f"{exp_name}_completed.txt")
    completed = load_completed(progress_file)

    logging.info(f"Backbone: {backbone}")
    logging.info(f"Running PCA + LDA (multiclass) for {len(languages)} languages")
    logging.info(f"Already completed: {sorted(list(completed))}")

    for lang in languages:
        if lang in completed:
            print(f"[SKIP] {lang} already completed.")
            continue

        print(f"\n[INFO] Running for language: {lang}")
        logging.info(f"----- Running for language: {lang} -----")

        # Update config for this run
        config["target"]["language"] = lang
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run the test with live stdout/stderr (so tqdm shows)
        process = subprocess.Popen(
            ["python", "-m", "models.Logistic.LogRegLDA"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line-buffered
        )

        # Stream output live and log
        with process.stdout:
            for line in process.stdout:
                print(line, end='')         # Show in terminal
                logging.info(line.strip())  # Log to file

        returncode = process.wait()

        if returncode == 0:
            logging.info(f"[SUCCESS] {lang} completed.")
            save_completed(progress_file, lang)
        else:
            logging.warning(f"[FAILURE] {lang} crashed or exited abnormally.")
            print(f"[ERROR] {lang} failed — check logs.")


if __name__ == "__main__":
    run_for_all_languages()
