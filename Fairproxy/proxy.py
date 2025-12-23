# Proxy identification & analysis

"""Proxy feature identification and analysis"""

import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency

class ProxyAnalyzer:
    """Analyze proxy features (movie, genre) correlation with gender"""
    
    def __init__(self, data):
        self.data = data
        
    def analyze_movie_proxy(self):
        """Analyze which movies are strong gender proxies"""
        # Movies watched by gender
        gender_movies = self.data.groupby(['clean_title', 'gender']).size().unstack(fill_value=0)
        
        # Chi-square test for independence
        results = []
        for movie in gender_movies.index:
            male_count = gender_movies.loc[movie, 'Male']
            female_count = gender_movies.loc[movie, 'Female']
            
            # Skip rare movies
            if male_count + female_count < 10:
                continue
            
            # Calculate chi-square
            contingency = [[male_count, female_count],
                          [self.data[self.data['gender']=='Male'].shape[0] - male_count,
                           self.data[self.data['gender']=='Female'].shape[0] - female_count]]
            chi2, p_value, _, _ = chi2_contingency(contingency)
            
            results.append({
                'movie': movie,
                'male_count': male_count,
                'female_count': female_count,
                'chi2': chi2,
                'p_value': p_value,
                'gender_ratio': male_count / (female_count + 1e-8)
            })
        
        return sorted(results, key=lambda x: x['chi2'], reverse=True)
    
    def analyze_genre_proxy(self):
        """Analyze which genres are strong gender proxies"""
        genre_gender = self.data.groupby(['main_genre', 'gender']).size().unstack(fill_value=0)
        
        results = []
        for genre in genre_gender.index:
            male_count = genre_gender.loc[genre, 'Male']
            female_count = genre_gender.loc[genre, 'Female']
            
            # Mutual information
            mi = mutual_info_score(
                [genre] * (male_count + female_count),
                ['Male'] * male_count + ['Female'] * female_count
            )
            
            results.append({
                'genre': genre,
                'male_count': male_count,
                'female_count': female_count,
                'mutual_info': mi,
                'gender_ratio': male_count / (female_count + 1e-8)
            })
        
        return sorted(results, key=lambda x: x['mutual_info'], reverse=True)
    
    def identify_strong_proxies(self, p_threshold=0.01):
        """Identify statistically significant proxies"""
        movie_proxies = self.analyze_movie_proxy()
        genre_proxies = self.analyze_genre_proxy()
        
        strong_movies = [m for m in movie_proxies if m['p_value'] < p_threshold]
        
        print(f"Strong movie proxies: {len(strong_movies)}")
        print(f"Top 5 genre proxies: {genre_proxies[:5]}")
        
        return {
            'movies': strong_movies,
            'genres': genre_proxies
        }