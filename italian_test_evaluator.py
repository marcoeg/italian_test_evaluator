"""
Italian Test Evaluator

This script automates the evaluation of Italian language proficiency using AI models.
It loads structured test data from a YAML file and queries different AI models (Ollama, OpenAI, DeepSeek, Anthropic Claude)
to generate responses and evaluate them based on predefined scoring criteria.

Main Features:

AI-powered response generation and scoring

Structured test data in YAML format

Multiple AI model integrations

Logging and error handling

CSV output for detailed results

Usage:
Run this script to execute the evaluation process and store results in a structured format.

Author: Marco Graziano. 
"""

import json
import yaml
import asyncio
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv
import os
import aiohttp
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import numpy as np  # Added for statistical calculations
from scoring import ItalianScoring  # Import the enhanced scoring system

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaConfig:
    """Configuration class for Ollama models."""
    def __init__(
        self,
        model: str = "llama3.2:latest",
        temperature: float = 0.7,
        context_length: int = 4096,
        num_predict: int = 128,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        system_prompt: Optional[str] = None
    ):
        self.model = model
        self.temperature = temperature
        self.context_length = context_length
        self.num_predict = num_predict
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.system_prompt = system_prompt

    def to_dict(self) -> Dict:
        """Convert config to Ollama API format."""
        config = {
            "model": self.model,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_length,
                "num_predict": self.num_predict,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty
            }
        }
        if self.system_prompt:
            config["system"] = self.system_prompt
        return config

class ItalianTestData:
    def __init__(self, yaml_file: str):
        """Initialize with test data from YAML file."""
        with open(yaml_file, 'r', encoding='utf-8') as f:
            self.test_data = yaml.safe_load(f)
        
        # Validate structure
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate the YAML structure matches expected format."""
        required_sections = [
            "Grammar & Verb Conjugation",
            "Vocabulary & Word Usage",
            "Reading Comprehension",
            "Conversational Skills",
            "Idioms & Expressions",
            "Cultural Competency",
            "Problem Solving"
        ]
        
        required_levels = ["Livello Base (A1-A2)", "Livello Intermedio (B1-B2)", "Livello Avanzato (C1-C2)"]
        
        for section in required_sections:
            if section not in self.test_data:
                raise ValueError(f"Missing required section: {section}")
            
            for level in required_levels:
                if level not in self.test_data[section]:
                    raise ValueError(f"Missing required level {level} in section {section}")

    def get_prompts(self) -> List[Dict]:
        """Get all prompts in a flat structure with metadata."""
        prompts = []
        for section, levels in self.test_data.items():
            for level, questions in levels.items():
                for q in questions:
                    prompts.append({
                        'section': section,
                        'level': level,
                        'prompt': q['prompt'],
                        'acceptable_answers': q['acceptable_answers'],
                        'criteria': q.get('criteria', []),
                        'max_score': q.get('max_score', 3)
                    })
        return prompts
    
class LLMEvaluator:
    def __init__(
        self, 
        api_keys: Dict[str, str], 
        ollama_base_url: str = "http://localhost:11434",
        ollama_configs: Optional[Dict[str, OllamaConfig]] = None
    ):
        """Initialize with API keys and configurations."""
        self.anthropic_client = AsyncAnthropic(api_key=api_keys.get("anthropic"))
        self.openai_client = AsyncOpenAI(api_key=api_keys.get("openai"))
        self.deepseek_token = api_keys.get("deepseek")
        self.ollama_base_url = ollama_base_url
        self.scoring_system = ItalianScoring(api_key=api_keys.get("openai"))  
        
        # Default Ollama configurations
        self.ollama_configs = ollama_configs or {
            "llama3.2:latest": OllamaConfig(
                model="llama3.2:latest",
                system_prompt="Sei un esperto della lingua italiana. Fornisci risposte concise e accurate, senza spiegazioni o contesto aggiuntivo."
            ),
            "mistral:latest": OllamaConfig(
                model="mistral:latest",
                temperature=0.8,
                system_prompt="Sei un esperto della lingua italiana. Fornisci risposte concise e accurate, senza spiegazioni o contesto aggiuntivo."
            ),
            "qwen2.5:latest": OllamaConfig(
                model="qwen2.5:latest",
                temperature=0.7,
                system_prompt="Sei un esperto della lingua italiana. Fornisci risposte concise e accurate, senza spiegazioni o contesto aggiuntivo."
            )
        }


    def create_evaluation_prompt(self, question: str, acceptable_answers: Dict[int, str], 
                               response: str, criteria: List[str], section: str, level: str) -> str:
        """Create evaluation prompt for GPT."""
        acceptable_answers_text = "\n".join([f"{score} punti: \"{answer}\"" for score, answer in acceptable_answers.items()])
        criteria_text = "\n".join([f"- {criterion}" for criterion in criteria])
        
        return f"""Agisci come un esperto di valutazione della lingua italiana. Valuta la seguente risposta:

Sezione: {section}
Livello: {level}
Domanda: {question}

Risposte accettabili:
{acceptable_answers_text}

Criteri di valutazione:
{criteria_text}

Risposta da valutare: "{response}"

Valuta la risposta considerando:
1. Accuratezza linguistica (grammatica, vocabolario)
2. Appropriatezza (registro, stile)
3. Completezza delle informazioni
4. Aderenza ai criteri di valutazione
5. Similarità con le risposte accettabili

Fornisci la valutazione in formato JSON:
{{
    "score": <punteggio_tra_0_e_{max(acceptable_answers.keys())}>,
    "reasons": [<lista_motivi_per_il_punteggio>],
    "feedback": "<feedback_specifico_per_miglioramenti>"
}}

Rispondi SOLO con il JSON, senza altro testo."""

    async def calculate_score(self, response: str, answer_data: Dict) -> Dict:
        """Calculate score using GPT for evaluation."""
        try:
            # Create evaluation prompt
            evaluation_prompt = self.create_evaluation_prompt(
                question=answer_data['prompt'],
                acceptable_answers=answer_data['acceptable_answers'],
                response=response,
                criteria=answer_data['criteria'],
                section=answer_data['section'],
                level=answer_data['level']
            )

            # Get GPT's evaluation
            eval_response = await self.scoring_client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "Sei un esperto di valutazione della lingua italiana."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3
            )

            # Parse evaluation
            evaluation = json.loads(eval_response.choices[0].message.content)
            
            return {
                "score": evaluation["score"],
                "max_score": answer_data['max_score'],
                "feedback": evaluation["feedback"],
                "reasons": evaluation["reasons"],
                "breakdown": {"detailed_score": evaluation["score"]}
            }

        except Exception as e:
            logger.error(f"Scoring error: {str(e)}")
            # Fallback to basic similarity scoring
            return self._fallback_scoring(response, answer_data)

    def _fallback_scoring(self, response: str, answer_data: Dict) -> Dict:
        """Fallback scoring method using basic similarity."""
        from difflib import SequenceMatcher
        
        # Get the highest similarity score among acceptable answers
        max_score = max(answer_data['acceptable_answers'].keys())
        best_similarity = max(
            SequenceMatcher(None, response.lower(), answer.lower()).ratio()
            for answer in answer_data['acceptable_answers'].values()
        )
        
        score = round(best_similarity * max_score, 2)
        
        return {
            "score": score,
            "max_score": max_score,
            "feedback": "Valutazione basata sulla similarità del testo",
            "reasons": ["Sistema di valutazione di fallback utilizzato"],
            "breakdown": {"similarity_score": best_similarity}
        }

    async def query_anthropic(self, prompt: str) -> str:
        """Query Claude API using official client."""
        try:
            message = await self.anthropic_client.messages.create(
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                model="claude-3-opus-20240229",
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    async def query_openai(self, prompt: str) -> str:
        """Query ChatGPT API using official client."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="o3-mini",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    async def query_deepseek(self, prompt: str) -> str:
        """Query DeepSeek API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_token}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"DeepSeek API error: {error_text}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise

    async def query_ollama(self, prompt: str, model_name: str = "llama3.2:latest") -> str:
        """Query local Ollama instance with specific model configuration."""
        if model_name not in self.ollama_configs:
            raise ValueError(f"Unsupported Ollama model: {model_name}")
        
        config = self.ollama_configs[model_name].to_dict()
        config["prompt"] = prompt
        config["stream"] = False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=config
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {error_text}")
                    
                    result = await response.json()
                    return result["response"]
        except Exception as e:
            logger.error(f"Ollama API error for model {model_name}: {str(e)}")
            raise

    async def list_available_ollama_models(self) -> List[str]:
        """List all available models in local Ollama instance."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {error_text}")
                    
                    result = await response.json()
                    return [model['name'] for model in result['models']]
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            raise

    async def  calculate_score(self, response: str, answer_data: Dict) -> Dict:
        """Calculate score using the enhanced scoring system."""
        reply = await self.scoring_system.score_response(response, answer_data)
        return reply

class TestRunner:
    def __init__(self, test_data: ItalianTestData, evaluator: LLMEvaluator):
        self.test_data = test_data
        self.evaluator = evaluator
        self.results = []

    async def run_test(self, llm_name: str, query_func) -> pd.DataFrame:
        """Run the full test suite against specified LLM."""
        prompts = self.test_data.get_prompts()
        
        for prompt_data in prompts:
            try:
                response = await query_func(prompt_data['prompt'])
                score_data = await self.evaluator.calculate_score(response, prompt_data)
                
                self.results.append({
                    'llm': llm_name,
                    'section': prompt_data['section'],
                    'level': prompt_data['level'],
                    'prompt': prompt_data['prompt'],
                    'response': response,
                    'expected_answer': prompt_data.get('acceptable_answers', {}),
                    'score': score_data['score'],
                    'max_score': score_data['max_score'],
                    'feedback': score_data['feedback'],
                    'score_breakdown': score_data.get('breakdown', {})
                })
                
            except Exception as e:
                logger.error(f"Error processing prompt {prompt_data['prompt']}: {str(e)}")
                continue
        
        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame with enhanced scoring details."""
        df = pd.DataFrame(self.results)
        
        # Add score percentage column
        df['score_percentage'] = (df['score'] / df['max_score'] * 100).round(2)
        
        # Convert score breakdown to separate columns if present
        if 'score_breakdown' in df.columns:
            breakdown_df = pd.json_normalize(df['score_breakdown'])
            df = pd.concat([df.drop('score_breakdown', axis=1), breakdown_df], axis=1)
        
        return df

    def save_results(self, output_dir: str):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results
        df = self.get_results_dataframe()
        df.to_csv(output_path / f"detailed_results_{timestamp}.csv", index=False)
        
        # Generate and save summary by section and level
        summary = df.groupby(['llm', 'section', 'level']).agg({
            'score': ['mean', 'sum', 'std'],
            'max_score': 'sum',
            'score_percentage': ['mean', 'std']
        }).round(2)
        
        summary.columns = ['avg_score', 'total_score', 'score_std', 'max_possible', 'avg_percentage', 'percentage_std']
        summary.to_csv(output_path / f"summary_{timestamp}.csv")
        
        # Save overall metrics with confidence intervals
        overall = df.groupby('llm').agg({
            'score': ['sum', 'mean', 'std'],
            'max_score': 'sum',
            'score_percentage': ['mean', 'std']
        })
        overall.columns = ['total_score', 'avg_score', 'score_std', 'max_possible', 'avg_percentage', 'percentage_std']
        
        # Calculate 95% confidence intervals
        n = len(df)
        confidence_interval = 1.96 * overall['score_std'] / np.sqrt(n)
        overall['confidence_interval'] = confidence_interval
        
        overall.to_csv(output_path / f"overall_{timestamp}.csv")

async def main():
    # Load test data
    test_data = ItalianTestData('italian_test.yaml')
    
    # Initialize evaluator with API keys from environment variables
    api_keys = {
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY")
    }
    
    # Custom Ollama configurations
    ollama_configs = {
        "llama3.2:latest": OllamaConfig(
            model="llama3.2:latest",
            temperature=0.7,
            context_length=4096,
            # system_prompt="You are an Italian language expert focusing on grammar, vocabulary, and cultural knowledge. Provide concise, accurate answers. Do not explain or provide additional context."
            system_prompt="Sei un esperto della lingua italiana, specializzato in grammatica, vocabolario e conoscenza culturale. Fornisci risposte concise e accurate, senza spiegazioni o contesto aggiuntivo.."
        ),
        "mistral:latest": OllamaConfig(
            model="mistral:latest",
            temperature=0.8,
            context_length=8192,
            system_prompt="Sei un esperto della lingua italiana, specializzato in grammatica, vocabolario e conoscenza culturale. Fornisci risposte concise e accurate, senza spiegazioni o contesto aggiuntivo.."

        ),
        "qwen2.5:latest": OllamaConfig(
            model="qwen2.5:latest",
            temperature=0.7,
            context_length=4096,
            system_prompt="Sei un esperto della lingua italiana, specializzato in grammatica, vocabolario e conoscenza culturale. Fornisci risposte concise e accurate, senza spiegazioni o contesto aggiuntivo.."
        ),
        "phi4:latest": OllamaConfig(
            model="phi4:latest",
            temperature=0.8,
            context_length=8192,
            system_prompt="Sei un esperto della lingua italiana, specializzato in grammatica, vocabolario e conoscenza culturale. Fornisci risposte concise e accurate, senza spiegazioni o contesto aggiuntivo.."

        ),
    }
    
    evaluator = LLMEvaluator(
        api_keys=api_keys,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_configs=ollama_configs
    )

    # List available Ollama models
    try:
        available_models = await evaluator.list_available_ollama_models()
        logger.info(f"Available Ollama models: {available_models}")
    except Exception as e:
        logger.warning(f"Could not list Ollama models: {str(e)}")
        available_models = []
    
    # Prepare model configurations for testing
    models = []
    
    # Add API-based models if keys are available
    if api_keys.get("anthropic"):
        models.append(("claude", evaluator.query_anthropic))
    if api_keys.get("openai"):
        models.append(("chatgpt", evaluator.query_openai))
    if api_keys.get("deepseek"):
        models.append(("deepseek", evaluator.query_deepseek))
    
    # Add available Ollama models
    ollama_model_names = ["llama3.2:latest", "mistral:latest", "qwen2.5:latest", "phi4:latest"]
    for model_name in ollama_model_names:
        if model_name in available_models:
            models.append(
                (f"ollama-{model_name}", 
                 lambda x, m=model_name: evaluator.query_ollama(x, model_name=m))
            )
    
    if not models:
        logger.error("No models available for testing. Please check API keys and Ollama installation.")
        return
    
    runner = TestRunner(test_data, evaluator)
    
    # Run tests for each model
    for model_name, query_func in models:
        logger.info(f"Running tests for {model_name}")
        try:
            await runner.run_test(model_name, query_func)
            logger.info(f"Completed testing {model_name}")
        except Exception as e:
            logger.error(f"Error running tests for {model_name}: {str(e)}")
            logger.error(f"Prompt Data: {json.dumps(prompt_data, indent=2, ensure_ascii=False)}")  # Log the full structure
            continue
    
    # Save all results
    logger.info("Saving test results...")
    runner.save_results("test_results")
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())