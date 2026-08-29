# ==============================================================================
# llm_engine.py — Multi-Provider LLM Interface
# ==============================================================================
# Supports:
#   1. Google Gemini (via google-generativeai)
#   2. GROQ (via groq sdk)
#   3. Local HuggingFace Model (via transformers pipeline)
#
# Unified interface via `generate_insight(prompt)` function.
# ==============================================================================

import os
import google.generativeai as genai
from groq import Groq
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from src.config import (
    LLM_PROVIDER,
    GEMINI_API_KEY, GEMINI_MODEL_NAME,
    GROQ_API_KEY, GROQ_MODEL_NAME,
    MODEL_DIR, LLM_MAX_TOKENS, LLM_TEMPERATURE
)
from src.logger import get_logger

logger = get_logger("LLMEngine")

# Singleton for local model
_local_pipeline = None


def _get_local_pipeline():
    """Load local HF model pipeline on demand."""
    global _local_pipeline
    if _local_pipeline is not None:
        return _local_pipeline

    logger.info("Loading local model from %s...", MODEL_DIR)
    try:
        if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
             raise FileNotFoundError(
                 f"Model directory {MODEL_DIR} is empty. "
                 "Please download a model there first."
             )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        _local_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=True
        )
        logger.info("Local model loaded successfully.")
        return _local_pipeline

    except Exception as e:
        logger.error("Failed to load local model: %s", e)
        raise


def _generate_gemini(prompt: str) -> str:
    """Generate text using Google Gemini."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing in .env")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            )
        )
        return response.text
    except Exception as e:
        logger.error("Gemini generation failed: %s", e)
        raise


def _generate_groq(prompt: str) -> str:
    """Generate text using GROQ."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in .env")

    client = Groq(api_key=GROQ_API_KEY)
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("GROQ generation failed: %s", e)
        raise


def _generate_local(prompt: str) -> str:
    """Generate text using local HF model."""
    pipe = _get_local_pipeline()
    try:
        # Format prompt for instruction-tuned models if necessary
        # Here we send raw prompt, assuming prompt template handles formatting
        output = pipe(prompt)
        generated_text = output[0]["generated_text"]
        
        # Strip the input prompt from the output if the model returns it
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
            
        return generated_text.strip()
    except Exception as e:
        logger.error("Local model generation failed: %s", e)
        raise


def generate_insight(prompt: str) -> str:
    """
    Generate insight using the configured LLM provider.

    Args:
        prompt: Full prompt string.

    Returns:
        Generated text response.
    """
    provider = LLM_PROVIDER
    logger.info("Generating insight with provider: %s", provider)

    try:
        if provider == "gemini":
            return _generate_gemini(prompt)
        elif provider == "groq":
            return _generate_groq(prompt)
        elif provider == "local":
            return _generate_local(prompt)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    except Exception as e:
        logger.error("Generation failed. Trying valid fallback message.")
        return f"[Error generating insight: {str(e)}. Please check logs and API keys.]"
