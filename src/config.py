import os
import torch
from pathlib import Path



from huggingface_hub import login
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# login into huggingface hub if HF_TOKEN is set in environment variables
hf_token = os.environ['HF_TOKEN']
login(token=hf_token)

# get the project root folder
project_root = Path(os.path.dirname(os.path.dirname(__file__)))

# PTB-XL dataset paths
ptb_xl_data_folder = project_root / "data" / "ptb-xl"
ptb_xl_transformed_data_folder = project_root / "data" / "ptb-xl-transformed"
processed_ptb_xl_data_folder = project_root / "data" / "ptb-xl-processed"

embeddings_folder = project_root / "data" / "embeddings"
embeddings_folder.mkdir(parents=True, exist_ok=True)

##################### . OLD CONFIG. KEEP FOR C-P
# Dataset ID and cache directory for the NIH Chest X-ray Pneumonia dataset
dataset_id = "orvile/brain-cancer-mri-dataset"
dataset_cache_directory = project_root / "data" / "bc_mri"


# Model ID,  cache directory for storing pre-trained models and fine-tuned versions
base_model_id = "google/medgemma-4b-it"
model_folder_base = project_root / "models", "medgemma-4b-it"
model_folder_bmri_ft_adapter = project_root / "models" / "medgemma-4b-it-brain-mri-adapter"
model_folder_bmri_ft_full = project_root / "models" / "medgemma-4b-it-nct-brain-mri-merged"
# Model loading keyword arguments
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# !!! ONLY FOR 4-BIT QUANTIZED MODELS !!!  Only use for high end GPUs
# which support bfloat16 compute in 4-bit quantization (e.g., NVIDIA H100, A100)

# model_kwargs["quantization_config"] = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
#     bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
# )

# prompt templates
condition_findings = [
    "A: brain glioma",
    "B: brain menin",
    "C: brain tumor",
]
condition_findings_str = "\n".join(condition_findings)
prompt_template = {
    "system_message": "You are a medical AI expert analyzing brain MRI images.",
    "user_prompt": f"What is the most likely type of brain cancer shown in the MRI image? \n {condition_findings_str}"
}

