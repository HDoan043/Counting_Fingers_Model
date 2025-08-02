from utils import loadModel, loadData
from model import predict, evaluate
import os

epochs = 2200
checkpoint_path = os.path.join("ckpt", f"{epochs}_epochs.pkl")
data_path = "data_for_count_model.csv"

# Load mô hình và data
model = loadModel(checkpoint_path)
_, testDataLoader = loadData(data_path)

# test
accuracy = evaluate(model, testDataLoader)
print(accuracy)
