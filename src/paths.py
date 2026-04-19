import os

BASE_DIR = os.path.dirname(__file__)

DATA = BASE_DIR +"/data"
DATA_TEMP = DATA +"/temp"
DATA_TEMP_IMG = DATA +"/temp_img"
MARKDOWNS = DATA +"/markdowns"
SPELLING = DATA +"/spelling"
SAMPLE = DATA +"/sample"

LLAMA_SERVER_URL = "http://localhost:8080/v1"
MODEL_DIR = os.getenv('MODEL_DIR')
QWEN35_9B = f"{MODEL_DIR}\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q8_0.gguf"  # 40-50 tokens/s
QWEN35_9B_MMPROJ = f"{MODEL_DIR}\Qwen3.5-9B-GGUF\mmproj-F16.gguf"
QWEN36_35B = f"{MODEL_DIR}\Qwen3.6-35B-A3B-GGUF\gemma-4-26B-A4B-it-UD-Q6_K.gguf" # 20-25 tokens/s
GEMMA4_26B = f"{MODEL_DIR}\gemma-4-26B-A4B-it-GGUF\Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf" # 20-40 tokens/s