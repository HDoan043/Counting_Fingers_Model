import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
from CountFingerModel import CountFingersModel

'''
    Hàm load dữ liệu từ file csv và tổ chức thành batch dữ liệu
'''
def loadData(root, ratio = 0.2):
    print("Start loading dataset...")
    # đọc dữ liệu từ file csv vào dataframe
    df = pd.read_csv(root)
    # loại bỏ dòng trống
    df = df.dropna(how='any')

    y = df.iloc[:, -1]

    # chia dữ liệu thành tập train và test theo tỉ lệ ratio
    df_train, df_test = train_test_split(df, test_size = ratio, random_state= 42, stratify= y, shuffle= True)

    # chia dữ liệu input và nhãn
    X_train = df_train.iloc[:, :-1].to_numpy() # input là tất cả, trừ cột cuối cùng
    y_train = df_train.iloc[:, -1].to_numpy() # label là cột cuối cùng
    X_test = df_test.iloc[:, :-1].to_numpy()
    y_test = df_test.iloc[:, -1].to_numpy()

    print(f"Input train: {X_train.shape}")
    print(f"Label test: {y_train.shape}")

    print(f"Input test: {X_test.shape}")
    print(f"Label test: {y_test.shape}")

    # Chuyển đổi input X và label y về tensor của pytorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Tạo dataloader
    trainSet = TensorDataset(X_train_tensor, y_train_tensor)
    testSet = TensorDataset(X_test_tensor, y_test_tensor)

    # Tạo dataloader
    trainDataloader = DataLoader(trainSet, batch_size = 8, shuffle = True)
    testDataLoader = DataLoader(testSet, batch_size = 8, shuffle = True)
    print("Finish load data!!!")
    return trainDataloader, testDataLoader

def loadModel(checkpoint_file):
    model = CountFingersModel()
    model.load_state_dict(torch.load(checkpoint_file))
    return model

def saveModel(model, checkpoint_file):
    with open(checkpoint_file, "w") as f:
        f.write("")

    torch.save(
        model.state_dict(),
        checkpoint_file
    )    