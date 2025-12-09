import os,json
 
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    # Ensure folders exist (only create if missing)
    for key in ["Input_Folder", "Output_Folder", "Log_Folder","Ocr_Folder","Ret_Folder"]:
        folder = config[key]
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")
 
    return config