from CountFingerModel import CountFingersModel
from utils import loadData, loadModel, saveModel
from model import train
import os



if __name__ == "__main__":
    restore_epochs = 1700
    if restore_epochs == 0:
        model = CountFingersModel()
    else:
        model = loadModel(os.path.join("ckpt", f"{restore_epochs}_epochs.pkl"))

    dataFile = "data_for_count_model.csv"
    trainDataLoader, testDataLoader = loadData(dataFile, ratio = 0.1)
    epochs = 500
    output = os.makedirs("ckpt", exist_ok = True)
    train(model, trainDataLoader, epochs)

    saveModel(model, os.path.join("ckpt", f"{restore_epochs + epochs}_epochs.pkl"))