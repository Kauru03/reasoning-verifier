import torch
import re
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    PeftModel
)
from src.logger import logger
from src.config import Config

class SmallerModelManager:
    """
    Manages the smaller model used as a verifier, including 
    LoRA fine-tuning and scoring logic.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.SMALLER_MODEL_BASE
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Initialized Smaller Model Manager for: {self.model_name}")

    def train_verifier(self, train_df, output_dir="verifier_lora"):
        """
        Executes LoRA training (Step 2 of the original pipeline).
        """
        from datasets import Dataset
        logger.info("Starting Smaller Model training (LoRA)...")

        # Prepare dataset
        dataset = Dataset.from_pandas(train_df)
        dataset = dataset.map(lambda r: {
            "text": f"\nQuestion:\n{r['input']}\n\nReasoning:\n{r['teacher_reasoning']}\n\nScore:\n{r['verifier_label']}\n"
        })

        # Load and prepare model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)

        # LoRA Configuration
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

        def tokenize(batch):
            t = self.tokenizer(batch["text"], padding="max_length", truncation=True, max_length=1024)
            t["labels"] = t["input_ids"].copy()
            return t

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format("torch")

        # Training process
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=2,
                per_device_train_batch_size=2,
                fp16=True,
                save_strategy="no",
                report_to="none"
            ),
            train_dataset=dataset
        )

        trainer.train()
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Smaller Model training complete. Saved to {output_dir}")

    def score_outputs(self, solver_df, lora_path="verifier_lora"):
        """
        Scores the solver's reasoning using the trained verifier (Step 4).
        """
        logger.info("Loading verifier for scoring...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        verifier = PeftModel.from_pretrained(base_model, lora_path).eval()

        scores = []
        for _, r in tqdm(solver_df.iterrows(), total=len(solver_df), desc="Scoring"):
            prompt = f"Evaluate the solution.\nReturn a score between 0.0 and 1.0.\n\nReasoning:\n{r['reasoning_model_llama']}\n\nFinal Answer:\n{r['reasoning_pred_answer_llama']}\n\nScore:\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(verifier.device)
            with torch.no_grad():
                out = verifier.generate(**inputs, max_new_tokens=6)

            decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # Guideline #8: Extracting score using regex from the original logic
            m = re.findall(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", decoded)
            scores.append(float(m[-1]) if m else 0.0)
            
        return scores