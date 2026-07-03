import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from itertools import product

import pandas as pd
import numpy as np
import joblib
import mlflow

from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier



mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("role_detection")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, data_path: str = "data/features.csv"):
        self.data_path = Path(data_path)
        self.model_dir = Path("models")
        self.report_dir = Path("reports")
        
        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_encoder_path = self.model_dir / "label_encoder.pkl"
        self.model_path = self.model_dir / "role_classifier.pkl"
        self.metrics_path = self.report_dir / "eval_metrics.json"
        
        self.meta_cols = ["meeting_id", "speaker_id", "role"]
        self.target_col = "role"

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads features, handles missing values, and fits LabelEncoder."""
        logger.info(f"Loading data from {self.data_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Identify feature columns
        self.feature_cols = [c for c in df.columns if c not in self.meta_cols]
        
        # Handle missing values in features
        df[self.feature_cols] = df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        
        # Label Encoding
        logger.info("Fitting LabelEncoder...")
        self.label_encoder = LabelEncoder()
        df["role_encoded"] = self.label_encoder.fit_transform(df[self.target_col].astype(str))
        
        # Save LabelEncoder
        joblib.dump(self.label_encoder, self.label_encoder_path)
        logger.info(f"Saved LabelEncoder to {self.label_encoder_path}")
        logger.info(f"Classes found: {list(self.label_encoder.classes_)}")
        
        return df

    def group_aware_split(self, df: pd.DataFrame):
        """Splits data into train/val/test (70/15/15) grouped by meeting_id."""
        logger.info("Performing group-aware split based on meeting_id (70/15/15)")
        meetings = df["meeting_id"].unique()
        
        # Ensure reproducibility
        np.random.seed(42)
        np.random.shuffle(meetings)
        
        n_meetings = len(meetings)
        n_train = max(int(0.70 * n_meetings), 1)
        n_val = max(int(0.15 * n_meetings), 1) if n_meetings >= 3 else 0
        
        train_meetings = meetings[:n_train]
        val_meetings = meetings[n_train:n_train + n_val]
        test_meetings = meetings[n_train + n_val:]
        
        train_df = df[df["meeting_id"].isin(train_meetings)].copy()
        val_df = df[df["meeting_id"].isin(val_meetings)].copy()
        test_df = df[df["meeting_id"].isin(test_meetings)].copy()
        
        logger.info(f"Split completed -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        self.X_train = train_df[self.feature_cols]
        self.y_train = train_df["role_encoded"].values
        
        self.X_val = val_df[self.feature_cols]
        self.y_val = val_df["role_encoded"].values
        
        self.X_test = test_df[self.feature_cols]
        self.y_test = test_df["role_encoded"].values
        
        self.test_df = test_df

    def _apply_data_augmentation(self) -> None:
        """
        Applies SMOTE to the training set to handle class imbalance at the data level.

        CRITICAL: This method ONLY touches self.X_train and self.y_train.
        Validation and test sets are never resampled.
        """
        from imblearn.over_sampling import SMOTE

        # Log distribution before resampling
        dist_before = Counter(self.y_train)
        dist_before_named = {
            self.label_encoder.inverse_transform([k])[0]: v
            for k, v in sorted(dist_before.items())
        }
        logger.info(f"Class distribution BEFORE SMOTE: {dist_before_named}")
        logger.info(f"Total training samples before: {len(self.y_train)}")

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)

        # Log distribution after resampling
        dist_after = Counter(y_resampled)
        dist_after_named = {
            self.label_encoder.inverse_transform([k])[0]: v
            for k, v in sorted(dist_after.items())
        }
        logger.info(f"Class distribution AFTER  SMOTE: {dist_after_named}")
        logger.info(f"Total training samples after:  {len(y_resampled)}")

        # Overwrite training data with the resampled arrays
        self.X_train = X_resampled
        self.y_train = y_resampled

    def _apply_thresholds(
        self,
        proba: np.ndarray,
        thresholds: np.ndarray
    ) -> np.ndarray:
        """
        Applies per-class probability thresholds to a probability matrix.

        For each sample, a class is considered eligible if its probability meets
        or exceeds the corresponding threshold. Among all eligible classes, the
        one with the highest probability is selected. If no class meets its
        threshold, the standard argmax class is returned as a fallback.

        Args:
            proba: Shape (n_samples, n_classes) probability matrix.
            thresholds: Shape (n_classes,) per-class threshold array.

        Returns:
            Array of predicted class indices of shape (n_samples,).
        """
        n_samples, n_classes = proba.shape

        # Boolean mask: which classes pass their threshold for each sample
        eligible_mask = proba >= thresholds  # (n_samples, n_classes)

        # Zero out probabilities for classes below threshold
        masked_proba = np.where(eligible_mask, proba, -1.0)

        # Argmax over the masked probabilities
        preds = np.argmax(masked_proba, axis=1)

        # Fallback: if ALL classes were masked out for a sample, use plain argmax
        no_eligible = ~eligible_mask.any(axis=1)
        if no_eligible.any():
            preds[no_eligible] = np.argmax(proba[no_eligible], axis=1)

        return preds

    def _optimize_thresholds(
        self,
        val_proba: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Grid-searches per-class probability thresholds on the validation set
        to maximize Macro F1-Score.

        The search iterates through thresholds from 0.10 to 0.90 (step 0.05)
        for every class independently. To keep this tractable for 4 classes
        (17^4 = 83,521 combinations), pure numpy broadcasting is used.

        Args:
            val_proba: Validation set probability matrix (n_samples, n_classes).
            y_val: True encoded labels for the validation set.

        Returns:
            Tuple of (best_thresholds array, best_macro_f1 score).
        """
        n_classes = val_proba.shape[1]
        candidate_thresholds = np.arange(0.10, 0.91, 0.05)

        logger.info(
            f"Starting threshold search: {len(candidate_thresholds)} candidates "
            f"per class, {n_classes} classes "
            f"({len(candidate_thresholds) ** n_classes:,} total combinations)"
        )

        best_f1 = -1.0
        best_thresholds = np.full(n_classes, 0.5)  # sensible default

        # Generate the full cartesian product of per-class thresholds
        for combo in product(candidate_thresholds, repeat=n_classes):
            t = np.array(combo)
            y_pred = self._apply_thresholds(val_proba, t)
            macro_f1 = f1_score(y_val, y_pred, average="macro")

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_thresholds = t.copy()

        logger.info(f"Threshold search complete. Best val Macro F1: {best_f1:.4f}")
        for i, cls_name in enumerate(self.label_encoder.classes_):
            logger.info(f"  {cls_name}: threshold = {best_thresholds[i]:.2f}")

        return best_thresholds, best_f1

    def train_and_evaluate(self):
        """Trains the XGBoost model within an MLflow run and evaluates it."""
        mlflow.set_experiment("role_detection")

        model_params = dict(
            objective="multi:softprob",
            num_class=len(self.label_encoder.classes_),
            max_depth=5,
            learning_rate=0.1,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            early_stopping_rounds=30
        )

        with mlflow.start_run():
            # Log params
            mlflow.log_params(model_params)

            # Apply SMOTE to training data ONLY (never val/test)
            self._apply_data_augmentation()

            logger.info("Initializing XGBoost classifier...")
            model = XGBClassifier(**model_params)

            logger.info("Training XGBoost on SMOTE-resampled data...")
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )

            # Save Model Local
            joblib.dump(model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")

            # Log Model to MLflow
            mlflow.xgboost.log_model(model, "xgboost_model")

            # ==============================================================
            # BASELINE EVALUATION (standard argmax)
            # ==============================================================
            logger.info("Evaluating BASELINE on test set (argmax)...")
            test_proba = model.predict_proba(self.X_test)
            y_pred_baseline = np.argmax(test_proba, axis=1)

            y_test_dec = self.label_encoder.inverse_transform(self.y_test)
            y_pred_baseline_dec = self.label_encoder.inverse_transform(y_pred_baseline)

            baseline_macro_f1 = f1_score(y_test_dec, y_pred_baseline_dec, average="macro")
            baseline_class_f1 = f1_score(
                y_test_dec, y_pred_baseline_dec,
                average=None, labels=self.label_encoder.classes_
            )

            logger.info(f"Baseline Macro F1: {baseline_macro_f1:.4f}")
            logger.info("\n" + classification_report(y_test_dec, y_pred_baseline_dec))

            mlflow.log_metric("baseline_macro_f1", baseline_macro_f1)
            for cls_name, f1 in zip(self.label_encoder.classes_, baseline_class_f1):
                mlflow.log_metric(f"baseline_f1_{cls_name}", float(f1))

            # ==============================================================
            # THRESHOLD OPTIMIZATION (post-processing on validation set)
            # ==============================================================
            logger.info("Running optimal threshold tuning on validation probabilities...")
            val_proba = model.predict_proba(self.X_val)
            best_thresholds, best_val_f1 = self._optimize_thresholds(val_proba, self.y_val)

            # Log optimized thresholds
            for i, cls_name in enumerate(self.label_encoder.classes_):
                mlflow.log_metric(f"opt_threshold_{cls_name}", float(best_thresholds[i]))

            # ==============================================================
            # TUNED EVALUATION (apply optimized thresholds to test set)
            # ==============================================================
            logger.info("Applying optimized thresholds to test set...")
            y_pred_tuned = self._apply_thresholds(test_proba, best_thresholds)
            y_pred_tuned_dec = self.label_encoder.inverse_transform(y_pred_tuned)

            tuned_macro_f1 = f1_score(y_test_dec, y_pred_tuned_dec, average="macro")
            tuned_class_f1 = f1_score(
                y_test_dec, y_pred_tuned_dec,
                average=None, labels=self.label_encoder.classes_
            )

            logger.info(f"Tuned Macro F1: {tuned_macro_f1:.4f}")
            logger.info("\n" + classification_report(y_test_dec, y_pred_tuned_dec))

            mlflow.log_metric("tuned_macro_f1", tuned_macro_f1)
            for cls_name, f1 in zip(self.label_encoder.classes_, tuned_class_f1):
                mlflow.log_metric(f"tuned_f1_{cls_name}", float(f1))

            # ==============================================================
            # ASSEMBLE & SAVE METRICS
            # ==============================================================
            improvement = tuned_macro_f1 - baseline_macro_f1
            logger.info(
                f"Threshold tuning {'improved' if improvement > 0 else 'did not improve'} "
                f"Macro F1 by {improvement:+.4f} "
                f"({baseline_macro_f1:.4f} -> {tuned_macro_f1:.4f})"
            )

            metrics: Dict[str, Any] = {
                "train_samples": int(len(self.X_train)),
                "val_samples": int(len(self.X_val)),
                "test_samples": int(len(self.X_test)),
                "baseline_macro_f1": float(baseline_macro_f1),
                "tuned_macro_f1": float(tuned_macro_f1),
                "improvement": float(improvement),
                "optimized_thresholds": {
                    cls_name: float(best_thresholds[i])
                    for i, cls_name in enumerate(self.label_encoder.classes_)
                },
            }

            for cls_name, b_f1, t_f1 in zip(
                self.label_encoder.classes_, baseline_class_f1, tuned_class_f1
            ):
                metrics[f"baseline_f1_{cls_name}"] = float(b_f1)
                metrics[f"tuned_f1_{cls_name}"] = float(t_f1)

            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved evaluation metrics to {self.metrics_path}")

def main():
    pipeline = TrainingPipeline()
    df = pipeline.load_and_preprocess()
    pipeline.group_aware_split(df)
    pipeline.train_and_evaluate()
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
