from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import platform
import psutil

# === Configuration ===

# Model name
#model_name = "codellama/CodeLlama-7b-hf"
model_name = "EleutherAI/gpt-j-6B"

# Text generation parameters
max_length = 250  # Maximum output length in tokens
temperature = 0.7  # Randomness level (lower = more predictable output)
top_p = 0.9  # Top-p sampling (probability threshold)
top_k = 50  # Top-k sampling (number of highest probability tokens)
do_sample = True  # Enable or disable randomness (True for more diverse output)

# Function to print current timestamp
def print_timestamp(label):
    print(f"{label}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === SYS and CUDA Diagnostics ===

def diagnostics():
    print("\nSystem and CUDA Diagnostics Info\n")
    
    # OS and Python Env
    print(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    from transformers import __version__ as transformers_version
    print(f"Transformers Version: {transformers_version}")
    
    # CPU and RAM 
    print(f"CPU Count: {psutil.cpu_count(logical=True)}")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    virtual_memory = psutil.virtual_memory()
    print(f"Total System RAM: {virtual_memory.total / 1e9:.2f} GB")
    print(f"Available System RAM: {virtual_memory.available / 1e9:.2f} GB")
    
    # GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Current Allocated Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Current Reserved Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max Reserved Memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("CUDA is not available. The model will run on CPU, which may be slower.")
    
# === Model Diagnostics ===

def model_diagnostics():
    print("\nModel-Specific Diagnostics Info\n")

    # Model-Specific
    print(f"Model name: {model_name}")
    print(f"Tokenizer Vocabulary Size: {tokenizer.vocab_size}")
    print("\nDetailed Model Configuration:")
    print(model.config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# Print starting timestamp
print_timestamp("Start")

# Print diagnostic information
diagnostics()

# === Load model ===

# Load tokenizer and model. `device_map="auto"` automatically distributes model across available devices.
print("\nLoading the model...")
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

# Print diagnostic information
model_diagnostics()

# === Input test prompt ===

# Text
#prompt = "I need a simple Python function to calculate the sum of two numbers. I only need the actual code."
prompt = "Explain Newton's First Law of Motion in simple terms with one clear example."

# Print the original prompt text
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
print()

# Print finishing timestamp
print_timestamp("End")
