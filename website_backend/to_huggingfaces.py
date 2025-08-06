from huggingface_hub import HfApi

model_folder = "deepseek_finetuned_lora_model"

repo_id = "auldo/cn_cultural_chatbot_v1"

api = HfApi()
api.upload_folder(
    folder_path=model_folder,
    repo_id=repo_id,
    repo_type="model",
    path_in_repo="", 
    commit_message="Initial upload of finetuned DeepSeek 7b model with LoRA weights",
)