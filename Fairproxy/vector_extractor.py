# Extract v0, v1, v2 from LLaMA

"""Extract internal LLaMA representation vectors"""

import torch
import numpy as np

class VectorExtractor:
    """Extract v0, v1, v2 from LLaMA internal layers"""
    
    def __init__(self, model, tokenizer, layer_idx=-2, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        
    def extract_all(self, movie_title, genre, predicted_gender):
        """
        Extract all three vectors for a sample
        
        Returns:
            {
                'v0': vector without gender (baseline),
                'v1': vector with predicted gender,
                'v2': vector with opposite gender
            }
        """
        opposite_gender = 'Female' if predicted_gender == 'Male' else 'Male'
        
        return {
            'v0': self.extract_v0(movie_title, genre),
            'v1': self.extract_v1(movie_title, genre, predicted_gender),
            'v2': self.extract_v2(movie_title, genre, opposite_gender)
        }
    
    def extract_v0(self, movie_title, genre):
        """Extract vector WITHOUT gender information (neutral context)"""
        prompt = f"""Based on the movie "{movie_title}" in the {genre} genre, recommend similar movies:"""
        return self._extract_hidden_state(prompt)
    
    def extract_v1(self, movie_title, genre, gender):
        """Extract vector WITH predicted gender"""
        prompt = f"""For a {gender} viewer who watched "{movie_title}" in the {genre} genre, recommend similar movies:"""
        return self._extract_hidden_state(prompt)
    
    def extract_v2(self, movie_title, genre, gender):
        """Extract vector WITH opposite gender"""
        prompt = f"""For a {gender} viewer who watched "{movie_title}" in the {genre} genre, recommend similar movies:"""
        return self._extract_hidden_state(prompt)
    
    def _extract_hidden_state(self, prompt):
        """Extract hidden state from target layer"""
        hidden_states = []
        
        def hook_fn(module, input, output):
            # output[0] is the hidden state tensor
            hidden_states.append(output[0].detach().cpu())
        
        # Register hook
        target_layer = self.model.model.layers[self.layer_idx]
        hook = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hook
        hook.remove()
        
        # Return last token's hidden state as numpy array
        # Shape: [batch_size, seq_len, hidden_dim] -> [hidden_dim]
        vector = hidden_states[0][0, -1, :].numpy()
        
        return vector
    
    def extract_batch(self, predictions):
        """Extract vectors for multiple predictions"""
        results = []
        
        for pred in predictions:
            vectors = self.extract_all(
                pred['movie_title'],
                pred['genre'],
                pred['predicted_gender']
            )
            
            results.append({
                'sample_id': pred['sample_id'],
                'movie_title': pred['movie_title'],
                'genre': pred['genre'],
                'predicted_gender': pred['predicted_gender'],
                'ground_truth': pred['ground_truth'],
                **vectors
            })
        
        return results