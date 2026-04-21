from src import Steam, train
from src.data_pre import data_pre

if __name__ == "__main__":
    S1 = Steam("train_data", "log")
    # 无需进行预处理
    train(S1)
    S2 = Steam("test_data", "log")
    data_pre(S2)