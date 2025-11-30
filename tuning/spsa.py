import subprocess
import random
import math
import time
import sys
import os

# Configuration
ENGINE_PATH = "../build/Release/rapfi.exe"  # Adjust path to your engine executable
PARAMS = [
    {"name": "ContinuationHistoryScale", "min": 0, "max": 4, "step": 1, "default": 1},
    {"name": "ContinuationHistory1PlyScale", "min": 0, "max": 4, "step": 1, "default": 1},
    {"name": "ContinuationHistoryMovePickScale", "min": 1, "max": 1024, "step": 10, "default": 256},
    {"name": "ContinuationHistory1PlyMovePickScale", "min": 1, "max": 1024, "step": 10, "default": 256},
]

# SPSA Hyperparameters
a = 2.0  # Step size scaling
c = 1.0  # Perturbation scaling
A = 100  # Stability constant
alpha = 0.602
gamma = 0.101

def run_match(params):
    """
    Runs a match (or a set of positions) with the given parameters.
    Returns a score (e.g., win rate or total score).
    For simplicity, this function currently runs a short self-play match.
    You should replace this with a proper match runner (e.g., using piskvork or a tournament manager).
    """
    # This is a placeholder. You need a way to evaluate the engine strength.
    # Ideally, run the engine against a fixed opponent or a pool of opponents.
    # Here we just simulate a score for demonstration.
    
    # In a real scenario, you would:
    # 1. Start the engine with the given params.
    # 2. Play a game against a reference engine.
    # 3. Return 1.0 for win, 0.5 for draw, 0.0 for loss.
    
    print(f"Simulating match with params: {params}")
    # Simulate a noisy objective function
    score = 0.5 
    # Add some fake dependency on params to show optimization (remove this in real usage)
    target_scale = 2
    score -= 0.1 * abs(params["ContinuationHistoryScale"] - target_scale)
    score += random.gauss(0, 0.05) # Noise
    return score

def spsa(max_iterations=100):
    current_params = {p["name"]: p["default"] for p in PARAMS}
    
    for k in range(1, max_iterations + 1):
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        
        delta = {p["name"]: random.choice([-1, 1]) for p in PARAMS}
        
        # Theta + ck * delta
        theta_plus = {}
        for p in PARAMS:
            name = p["name"]
            val = current_params[name] + ck * delta[name]
            theta_plus[name] = max(p["min"], min(p["max"], val)) # Clip
            
        # Theta - ck * delta
        theta_minus = {}
        for p in PARAMS:
            name = p["name"]
            val = current_params[name] - ck * delta[name]
            theta_minus[name] = max(p["min"], min(p["max"], val)) # Clip
            
        y_plus = run_match(theta_plus)
        y_minus = run_match(theta_minus)
        
        # Update parameters
        ghat = (y_plus - y_minus) / (2 * ck)
        
        print(f"Iter {k}: y+={y_plus:.4f}, y-={y_minus:.4f}, ghat={ghat:.4f}")
        
        for p in PARAMS:
            name = p["name"]
            update = ak * ghat * delta[name]
            current_params[name] += update
            current_params[name] = max(p["min"], min(p["max"], current_params[name])) # Clip
            
        print(f"Current Params: {current_params}")

if __name__ == "__main__":
    print("Starting SPSA Tuning...")
    spsa()
