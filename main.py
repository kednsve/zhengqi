from src import Steam, train

if __name__ == "__main__":
    S = Steam("train_data", "log")
    # 无需进行预处理
    train(S)
