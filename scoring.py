"""
Italian Language Response Scoring

This module provides the scoring mechanism for evaluating AI-generated responses.
It uses OpenAI API for primary evaluation and a fallback similarity-based method
if AI-based scoring fails.

Features:

AI-driven evaluation via OpenAI

Fallback similarity scoring using text comparison

JSON-based structured response format

Usage:
Called within italian_test_evaluator.py to assign scores and feedback to AI-generated answers.

Author: Marco Graziano.
"""

from typing import Dict, List
import json
from openai import AsyncOpenAI

class ItalianScoring:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    def create_evaluation_prompt(self, question: str, section: str, level: str, 
                               acceptable_answers: Dict[int, str], response: str, 
                               criteria: List[str]) -> str:
        """Create evaluation prompt for GPT."""
        return f"""Agisci come un esperto di linguistica italiana e valuta la seguente risposta.

Sezione: {section}
Livello: {level}
Domanda: {question}

Risposte accettabili per punteggio:
{chr(10).join([f'{score} punto/i: "{answer}"' for score, answer in acceptable_answers.items()])}

Criteri di valutazione:
{chr(10).join([f'- {criterion}' for criterion in criteria])}

Risposta da valutare:
"{response}"

Fornisci la valutazione in formato JSON seguendo questo schema:
{{
    "score": <punteggio_numerico_tra_0_e_{max(acceptable_answers.keys())}>,
    "reasons": [<lista_di_motivi_per_il_punteggio>],
    "feedback": "<feedback_specifico_per_miglioramenti>"
}}

Fornisci la valutazione in formato JSON valido e preciso. Assicurati di valutare la risposta considerando anche variazioni lessicali, sinonimi e piccole differenze grammaticali.  

Restituisci SOLO il JSON, senza testo aggiuntivo. """

    async def score_response(self, response: str, answer_data: Dict) -> Dict:
        """Score a response using GPT."""
        try:
            # Create evaluation prompt
            prompt = self.create_evaluation_prompt(
                question=answer_data['prompt'],
                section=answer_data['section'],
                level=answer_data['level'],
                acceptable_answers=answer_data['acceptable_answers'],
                response=response,
                criteria=answer_data.get('criteria', [])
            )

            # Get GPT's evaluation
            completion = await self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "Sei un esperto linguista italiano. Valuta le risposte in base ai criteri forniti."},
                    {"role": "user", "content": prompt}
                ],
            )

            # Parse GPT's response
            evaluation = json.loads(completion.choices[0].message.content)

            return {
                "score": evaluation["score"],
                "max_score": max(answer_data['acceptable_answers'].keys()),
                "feedback": evaluation["feedback"],
                "reasons": evaluation["reasons"]
            }

        except Exception as e:
            print(f"Error in scoring: {str(e)}")
            # Fallback to basic similarity scoring if GPT evaluation fails
            return self._fallback_scoring(response, answer_data)

    def _fallback_scoring(self, response: str, answer_data: Dict) -> Dict:
        """Fallback scoring method using basic similarity."""
        from difflib import SequenceMatcher
        
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
            "reasons": ["Utilizzato sistema di valutazione di fallback"]
        }