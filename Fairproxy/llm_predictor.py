# LLaMA gender prediction

"""LLaMA-based gender prediction from (movie, genre)"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class LLaMAGenderPredictor:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
    def predict_gender(self, movie_title, genre):
        """
        Predict gender from movie and genre
        
        Returns:
            {
                'predicted_gender': 'Male'/'Female',
                'confidence': float,
                'logits': tensor
            }
        """
        prompt = self._build_prompt(movie_title, genre)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Get probabilities for "Male" and "Female" tokens
        male_token_id = self.tokenizer.encode("Male", add_special_tokens=False)[0]
        female_token_id = self.tokenizer.encode("Female", add_special_tokens=False)[0]
        
        probs = torch.softmax(logits, dim=0)
        male_prob = probs[male_token_id].item()
        female_prob = probs[female_token_id].item()
        
        # Normalize
        total = male_prob + female_prob
        male_prob /= total
        female_prob /= total
        
        if male_prob > female_prob:
            return {
                'predicted_gender': 'Male',
                'confidence': male_prob,
                'male_prob': male_prob,
                'female_prob': female_prob
            }
        else:
            return {
                'predicted_gender': 'Female',
                'confidence': female_prob,
                'male_prob': male_prob,
                'female_prob': female_prob
            }
    
    def _build_prompt(self, movie_title, genre):
        """Build prediction prompt"""
        return f"""A person watched the movie "{movie_title}" which is in the {genre} genre.
Based on typical viewing patterns, is this person more likely Male or Female?
Answer with just one word: Male or Female.
Answer:"""
    
    def predict_batch(self, data, confidence_threshold=0.8):
        """Predict for multiple samples and filter by confidence"""
        results = []
        
        for idx, row in data.iterrows():
            pred = self.predict_gender(row['clean_title'], row['main_genre'])
            
            # Add ground truth
            pred['ground_truth'] = row['gender']
            pred['correct'] = (pred['predicted_gender'] == pred['ground_truth'])
            pred['movie_title'] = row['clean_title']
            pred['genre'] = row['main_genre']
            pred['sample_id'] = idx
            
            # Filter by confidence
            if pred['confidence'] >= confidence_threshold:
                results.append(pred)
        
        print(f"High confidence predictions: {len(results)}/{len(data)}")
        print(f"Accuracy: {sum(r['correct'] for r in results)/len(results):.3f}")
        
        return results