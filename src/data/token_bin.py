from pathlib import Path


def get_token_bin_data(train_path, val_path):
    train_path = Path(train_path)
    val_path = Path(val_path)
    missing = [str(path) for path in (train_path, val_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing pre-tokenized uint16 data files: " + ", ".join(missing)
        )
    return {"train": str(train_path), "val": str(val_path)}
