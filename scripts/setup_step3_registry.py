"""Script to execute Step 03 Model Registry Walkthrough.

1. Trains and logs the best model (XGBoost Regressor) -> registers as 'ride-duration-predictor' v1.
2. Walks v1 through lifecycle: None -> Staging -> Production.
3. Trains and logs a worse model (Linear Regression) -> registers as 'ride-duration-predictor' v2.
4. Leaves v2 in None stage for contrast.
"""

import logging
import time

from prodml.config import get_settings
from prodml.data import clean_data, load_data, split_data
from prodml.features import ORDERED_FEATURE_NAMES, features_to_matrix, prepare_features
from prodml.registry import (
    get_client,
    get_production_model_version,
    register_candidate_model,
    transition_stage,
)
from prodml.train import (
    compute_file_sha256,
    get_git_commit_hash,
    train_linear_regression,
    train_xgboost,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("step3_setup")


def main():
    settings = get_settings()
    client = get_client()
    model_name = "ride-duration-predictor"

    logger.info("Loading dataset from %s", settings.data_path)
    raw_df = load_data(settings.data_path)
    cleaned_df = clean_data(raw_df)
    X_train_df, X_val_df, _, y_train, y_val, _ = split_data(cleaned_df)

    X_train = features_to_matrix(prepare_features(X_train_df), ORDERED_FEATURE_NAMES)
    X_val = features_to_matrix(prepare_features(X_val_df), ORDERED_FEATURE_NAMES)
    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    git_commit = get_git_commit_hash()
    data_hash = compute_file_sha256(settings.data_path)
    data_version = data_hash[:8]

    # -------------------------------------------------------------
    # 1. Train & Log Version 1: Best Model (XGBoost Baseline)
    # -------------------------------------------------------------
    logger.info("--- Step 1: Training Version 1 Candidate (XGBoost) ---")
    xgb_metrics, xgb_model = train_xgboost(
        X_train=X_train,
        y_train=y_train_arr,
        X_val=X_val,
        y_val=y_val_arr,
        settings=settings,
        git_commit=git_commit,
        data_version=data_version,
        data_hash=data_hash,
        run_name="xgboost-best-v1",
        enable_autolog=False,
    )
    import mlflow

    xgb_run = mlflow.last_active_run()
    xgb_run_id = xgb_run.info.run_id
    logger.info(
        "XGBoost Run complete: run_id=%s, MAE=%.4f", xgb_run_id, xgb_metrics["mae"]
    )

    # Register as Version 1
    v1 = register_candidate_model(
        run_id=xgb_run_id, model_name=model_name, client=client
    )
    logger.info(
        "Registered Version 1: version=%s, stage=%s", v1.version, v1.current_stage
    )

    # -------------------------------------------------------------
    # 2. Lifecycle Transition: Version 1 -> Staging -> Production
    # -------------------------------------------------------------
    logger.info("--- Step 2: Transitioning Version 1 -> Staging ---")
    v1_staging = transition_stage(
        model_name=model_name, version=v1.version, stage="Staging", client=client
    )
    logger.info("Version 1 stage is now: %s", v1_staging.current_stage)
    time.sleep(1)

    logger.info("--- Step 3: Transitioning Version 1 -> Production ---")
    v1_prod = transition_stage(
        model_name=model_name,
        version=v1.version,
        stage="Production",
        archive_existing=True,
        client=client,
    )
    logger.info("Version 1 stage is now: %s", v1_prod.current_stage)

    # -------------------------------------------------------------
    # 3. Train & Log Version 2: Worse Model (Linear Regression)
    # -------------------------------------------------------------
    logger.info("--- Step 4: Training Version 2 Candidate (Linear Regression) ---")
    lr_metrics = train_linear_regression(
        X_train=X_train,
        y_train=y_train_arr,
        X_val=X_val,
        y_val=y_val_arr,
        settings=settings,
        git_commit=git_commit,
        data_version=data_version,
        data_hash=data_hash,
    )
    lr_run = mlflow.last_active_run()
    lr_run_id = lr_run.info.run_id
    logger.info(
        "Linear Regression Run complete: run_id=%s, MAE=%.4f",
        lr_run_id,
        lr_metrics["mae"],
    )

    # Register as Version 2 and leave in None
    v2 = register_candidate_model(
        run_id=lr_run_id, model_name=model_name, client=client
    )
    logger.info(
        "Registered Version 2: version=%s, stage=%s (left in None for contrast)",
        v2.version,
        v2.current_stage,
    )

    # -------------------------------------------------------------
    # Summary of Registry State
    # -------------------------------------------------------------
    logger.info("--- Model Registry Summary for '%s' ---", model_name)
    all_versions = client.search_model_versions(f"name = '{model_name}'")
    for ver in all_versions:
        logger.info(
            "Version %s | Stage: %-12s | Run ID: %s | Description: %s",
            ver.version,
            ver.current_stage,
            ver.run_id,
            ver.description,
        )

    prod = get_production_model_version(model_name, client)
    logger.info(
        "Active Production Model: Version %s (Run %s)", prod.version, prod.run_id
    )


if __name__ == "__main__":
    main()
