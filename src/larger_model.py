from openai import AzureOpenAI
from src.config import Config
from src.logger import logger

class LargerModelClient:
    """
    Handles interactions with the high-capacity model (e.g., GPT-4) 
    for generating reasoning and gold-standard data.
    """
    
    def __init__(self):
        
        if not Config.AZURE_API_KEY or not Config.AZURE_ENDPOINT:
            logger.error("Azure OpenAI credentials missing in .env file")
            raise ValueError("Missing Azure Credentials")

        self.client = AzureOpenAI(
            api_key=Config.AZURE_API_KEY,
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_version=Config.AZURE_VERSION
        )
        
        logger.info("Larger Model Client successfully initialized.")

    def generate_response(self, problem: str, deployment_name: str = None):
        """
        Queries the larger model to solve a problem step-by-step.
        """
        deployment = deployment_name or Config.LARGER_MODEL_NAME
        
        prompt_tmpl = (
            "Solve step by step.\n"
            "End with:\n"
            "Final Answer: <answer>\n\n"
            "Problem:\n{problem}"
        )
        
        prompt = prompt_tmpl.format(problem=problem)
        
        try:
            logger.debug(f"Sending request to Larger Model ({deployment})")
            
            resp = self.client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048
            )
            return resp.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error during Larger Model inference: {str(e)}")
            return None