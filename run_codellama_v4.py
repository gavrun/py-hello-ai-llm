from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import platform

# === Configuration ===

# Model name
model_name = "codellama/CodeLlama-7b-hf"

# Text generation parameters
max_length = 1000  # Maximum output length in tokens
temperature = 0.7  # Randomness level (lower = more predictable output)
top_p = 0.9  # Top-p sampling (probability threshold)
top_k = 50  # Top-k sampling (number of highest probability tokens)
do_sample = True  # Enable or disable randomness (True for more diverse output)

# Function to print current timestamp
def print_timestamp(label):
    print(f"{label}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === CUDA Diagnostics ===

def cuda_diagnostics():
    print("\nCUDA Diagnostics\n")
    print(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    from transformers import __version__ as transformers_version
    print(f"Transformers Version: {transformers_version}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("CUDA is not available. The model will run on CPU, which may be slower.")
    print(f"\nModel name: {model_name}")

# Print starting timestamp
print_timestamp("Start")

# Print CUDA diagnostics
cuda_diagnostics()

# === Load the model ===

# Load tokenizer and model. `device_map="auto"` automatically distributes model across available devices.
print("Loading the model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="./offload",
    torch_dtype=torch.float16
)
print("Model loaded successfully.")
print(f"Model loaded on device: {model.device}")
if torch.cuda.is_available():
    print(f"Post-load GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Post-load GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Print timestamp after model loading
print_timestamp("Model Loaded")

# === Input prompt ===

# Input text
prompt = "I need a simple Python function to calculate the sum of two numbers."

# Print the original prompt
print("\nOriginal Prompt:")
print(prompt)

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
print("\nGenerating response...")
try:
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id, # Pad with the end-of-sequence token if necessary
        no_repeat_ngram_size=2 # Avoid repeating the same phrases
    )
    # Decode the result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
except Exception as e:
    print(f"An error occurred during generation: {e}")
    result = None

# Print the generated response
print("\nGenerated Response:")
print(result if result else "No response generated due to an error.")

# Print finishing timestamp
print_timestamp("End")
