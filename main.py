import argparse
import pandas as pd
from src.logger import logger
from src.config import Config
from src.data_utils import DataHandler
from src.larger_model import LargerModelClient
from src.smaller_model import SmallerModelManager
from src.evaluator import Evaluator

def run_pipeline(args):
    """
    Main orchestration logic for the Reasoning Verifier.
    Follows the 6-step pipeline while respecting cached files.
    """
    # Initialize Handlers
    data_handler = DataHandler()
    larger_model = LargerModelClient()
    smaller_model = SmallerModelManager(args.verifier_base)
    evaluator = Evaluator()

    # --- STEP 1: Generate/Load Verifier Training Data ---
    verifier_train_path = "verifier_train.jsonl"
    if not data_handler.check_exists(verifier_train_path):
        logger.info("Generating training data using Larger Model...")
        train_df = data_handler.load_jsonl(args.train_data)
        if args.max_train:
            train_df = train_df.head(args.max_train)
        
        rows = []
        for _, row in train_df.iterrows():
            reasoning = larger_model.generate_response(row["problem"], args.larger_model)
            rows.append({
                "input": row["problem"],
                "teacher_reasoning": reasoning,
                "verifier_label": 1.0
            })
        
        teacher_df = pd.DataFrame(rows)
        data_handler.save_jsonl(teacher_df, verifier_train_path)
    else:
        teacher_df = data_handler.load_jsonl(verifier_train_path)

    # --- STEP 2: Train Smaller Model (Verifier) ---
    if not data_handler.check_exists("verifier_lora"):
        smaller_model.train_verifier(teacher_df, output_dir="verifier_lora")

    # --- STEP 3 & 4: Load Solver Data and Score ---
    scored_path = "verifier_scored_testset.jsonl"
    if not data_handler.check_exists(scored_path):
        solver_df = data_handler.load_jsonl("solver_outputs.jsonl")
        if solver_df is None:
            logger.error("Solver outputs missing. Step 3 & 4 cannot proceed.")
            return
            
        scores = smaller_model.score_outputs(solver_df, lora_path="verifier_lora")
        solver_df["verifier_score"] = scores
        data_handler.save_jsonl(solver_df, scored_path)
    else:
        solver_df = data_handler.load_jsonl(scored_path)

    # --- STEP 5: Evaluation ---
    metrics = evaluator.calculate_gating_metrics(solver_df, args.threshold)
    pd.DataFrame([metrics]).to_csv("verifier_gating_metrics.csv", index=False)
    logger.info(f"Final Metrics saved. Accuracy: {metrics['accuracy']:.4f}")

    # --- STEP 6: Reasoning Similarity ---
    test_df = data_handler.load_jsonl(args.test_data)
    if args.max_test:
        test_df = test_df.head(args.max_test)
        
    similarities = evaluator.compute_reasoning_similarity(test_df, solver_df, args.larger_model)
    solver_df["reasoning_similarity"] = similarities
    data_handler.save_jsonl(solver_df, "final_results_all_steps.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reasoning Verifier CLI")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--larger_model", default=Config.LARGER_MODEL_NAME)
    parser.add_argument("--verifier_base", default=Config.SMALLER_MODEL_BASE)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)

    args = parser.parse_args()
    run_pipeline(args)