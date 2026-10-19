APP_NAME = "penalty-shape"
MODEL = "Qwen/Qwen3-8B"
MONITOR_MODEL = "claude-haiku-5-5"
R_H = 0.5
FLAG_THRESHOLD = 3
PROMPT_TOKENS = 1024
RESPONSE_TOKENS = 8192
TEMPERATURE = 0.8
VERL_COMMIT = "8f8b122195ecf43475679fb6ee0204ee33f387c5"
VERL_IMAGE = "verlai/verl:vllm024.dev2"
CELLS = {
    "p150_seq": {"p": 1.5, "credit": "sequence"},
    "p025_token": {"p": 0.25, "credit": "token"},
}
N_CALIBRATION = 64
N_DEV = 64
SPLIT_SEED = 0
