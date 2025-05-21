from datasets import load_dataset


# Defining names and saving paths of datasets
datasets_paths = {
    "wikitext-103-raw-v1": "./Datasets/wikitext-103",
    "wikitext-2-raw-v1": "./Datasets/wikitext-2",
    "squad_v2": "./Datasets/squad_v2",
}

# Downloading and saving datasets
for config_name, save_path in datasets_paths.items():
    try:
        if "wikitext" in config_name:
            dataset = load_dataset("wikitext", config_name)
        else:
            dataset = load_dataset(config_name)
        dataset.save_to_disk(save_path)
        print(f"✅ Dataset '{config_name}' saved in '{save_path}'")
    except Exception as e:
        print(f"❌ Error during processing '{config_name}': {e}")
