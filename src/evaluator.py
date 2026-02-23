import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.logger import logger
from src.larger_model import LargerModelClient

class Evaluator:
    """
    Handles the evaluation of the verifier's accuracy and 
    the similarity between Larger Model and Smaller Model reasoning.
    """

    def __init__(self):
        # Guideline #5: Logging instead of printing
        logger.info("Initializing Evaluator and loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.larger_model = LargerModelClient()

    def calculate_gating_metrics(self, solver_df, threshold: float):
        """
        Calculates Accuracy, TP, FP, TN, FN based on the verifier threshold.
        """
        logger.info(f"Evaluating gating performance at threshold: {threshold}")

        solver_df["solver_correct"] = (
            solver_df["reasoning_pred_answer_llama"].astype(str).str.strip()
            == solver_df["answer"].astype(str).str.strip()
        ).astype(int)

        solver_df["verifier_accept"] = (
            solver_df["verifier_score"] >= threshold
        ).astype(int)

        conf = confusion_matrix(
            solver_df["solver_correct"],
            solver_df["verifier_accept"]
        )
        
        tn, fp, fn, tp = conf.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        metrics = {
            "threshold": threshold,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }
        
        logger.info(f"Gating Accuracy: {accuracy:.4f}")
        return metrics

    def compute_reasoning_similarity(self, test_df, solver_df, deployment_name: str):
        """
        Compares Larger Model reasoning vs. Solver reasoning using cosine similarity.
        """
        logger.info("Generating Larger Model reasoning for test set similarity check...")
        
        teacher_reasons = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Larger Model Inference"):
            reasoning = self.larger_model.generate_response(row["problem"], deployment_name)
            teacher_reasons.append(reasoning if reasoning else "")

        solver_df["larger_model_test_reasoning"] = teacher_reasons

        logger.info("Computing cosine similarity scores...")
        similarities = []
        for s, t in zip(solver_df["reasoning_model_llama"], solver_df["larger_model_test_reasoning"]):
            if not s or not t:
                similarities.append(0.0)
                continue
                
            sim = torch.cosine_similarity(
                self.embedder.encode(s, convert_to_tensor=True),
                self.embedder.encode(t, convert_to_tensor=True),
                dim=0
            ).item()
            similarities.append(sim)

        return similarities