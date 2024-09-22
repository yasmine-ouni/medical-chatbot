import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Starting the script...")  # Debugging statement

# Update this to the local path where the model is stored
model_directory = "D:\\inotequia\\med2\\TinyLlama"  # Local path to the downloaded model

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")

try:
    print("Loading tokenizer...")  # Debugging statement
    tokenizer = AutoTokenizer.from_pretrained(model_directory)  # Load tokenizer from local directory
    print("Tokenizer loaded successfully!")  # Debugging statement
    
    print("Loading model...")  # Debugging statement
    model = AutoModelForCausalLM.from_pretrained(model_directory)  # Load model from local directory
    print("Model loaded successfully!")  # Debugging statement
    
    # Move model to the appropriate device
    model.to(device)
    print("Model moved to device successfully!")  # Debugging statement
except Exception as e:
    print(f"Error loading model: {e}")