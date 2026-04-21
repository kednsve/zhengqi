from pathlib import Path

config = {
    "log_fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "root_path": Path(__file__).parent.parent,
    "data_path": Path(__file__).parent.parent / "data",
    "test_data": Path(__file__).parent.parent / "data" / "zhengqi_test.txt",
    "train_data": Path(__file__).parent.parent / "data" / "zhengqi_train.txt",
    "model_path": Path(__file__).parent.parent / "model",
    "json_data": Path(__file__).parent.parent / "data" / "data.json",
}
for i in config.keys():
    if i.endswith("path"):
        config[i].mkdir(parents=True, exist_ok=True)
