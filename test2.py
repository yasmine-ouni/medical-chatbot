from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
model_directory = "D:\\inotequia\\med2\\TinyLlama"  # Update this path as necessary

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Create pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.8,
    max_new_tokens=50,
    do_sample=True,
    device=-1  # Use -1 for CPU
)

# Test prompt
prompt = "You are an interactive virtual medical assistant. Your goal is to provide accurate, empathetic, and engaging responses.\n\nUser's Question: \"hi\"\n\nContext Information: \"No relevant medical history provided.\"\n\nRespond directly and engagingly to the user's question:"

response = llm_pipeline(prompt)
print(response)
