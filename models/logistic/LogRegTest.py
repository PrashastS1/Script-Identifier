import os
import yaml
import subprocess
import logging


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


def run_for_all_languages():
    # Load config
    config_path = "conifg/logreg.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_args = config["dataset"]
    logreg_cfg = config["logreg_params"]
    exp_name = logreg_cfg.get("exp_name", "logreg_batch")

    # Load language mapping
    with open('./dataset/language_encode.json') as f:
        lang_map = yaml.safe_load(f)

    languages = list(lang_map.keys())

    # Setup batch logger
    log_path = setup_logger("logs", exp_name)
    print(f"[INFO] Batch logging to: {log_path}")

    logging.info(f"Backbone: {dataset_args.get('backbone', 'unknown')}")
    logging.info("Experiment Type: PCA + LDA (multiclass)")
    logging.info(f"Languages to evaluate: {languages}")

    for lang in languages:
        # Update the YAML config
        config["target"]["language"] = lang
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        print(f"\n[INFO] Running experiment for language: {lang}")
        logging.info(f"----- Running for language: {lang} -----")

        # Call LogRegLDA.py as a module
        result = subprocess.run(
            ["python", "-m", "models.Logistic.LogRegLDA"],
            capture_output=True,
            text=True
        )

        # Log output
        logging.info(result.stdout)
        logging.error(result.stderr if result.stderr else "No stderr")

        if result.returncode != 0:
            logging.warning(f"[WARNING] Run failed for language: {lang}")
        else:
            logging.info(f"[INFO] Finished run for language: {lang}")


if __name__ == "__main__":
    run_for_all_languages()
