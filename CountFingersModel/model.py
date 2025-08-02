import torch

def train(model, dataloader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)

    loss = torch.nn.CrossEntropyLoss().to(device)
    lr = 0.0004
    optimizer = torch.optim.Adam(model.parameters(), lr)
    l = len(dataloader)

    
    for i in range(epochs):
        print()
        print(f"-------------------- EPOCH: {i} --------------------")
        print()
        index = 1
        mse_loss = 0 # Tính trung bình của bình phương loss của tất cả các batch
        for data, label in dataloader:
            print(f"\rEPOCH {i}, Step {index}/{l}", end="")
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            # tính toán hàm loss
            loss_function = loss(output, label)

            # xóa gradient về 0
            optimizer.zero_grad()

            mse_loss += torch.square(loss_function)
        
            loss_function.backward()
            # cập nhật trọng số
            optimizer.step()
            index+=1

        print()
        # Cập nhật lại giá trị trung bình của bình phương loss theo cả batch sau mỗi lần duyệt xong tất cả các batch
        #  để có tính tổng quát hóa
        print(f"Loss: {torch.sqrt(mse_loss/l)}")   
        

def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model = model.to(device)

    # Duyệt qua dataloader
    correct = 0
    total = 0
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)

        total += len(data)

        # Chạy suy diễn cho 1 batch 
        output_batch = model(data)
        # output_batch có kích thước là số_lớp x số_mẫu_mỗi_batch
        # Cần lấy ra chỉ số của phần tử có xác suất lớn nhất dữ đoán được trong batch
        output_predict = torch.argmax(output_batch, dim = 1) # output_predict có kích thước là 1 x số_mẫu_mỗi_batch
    
        # So sánh với nhãn ground-truth
        result = ( output_predict == label )
        correct += torch.sum(result).item()

    return correct/total

def predict(model, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model = model.to(device)
    
    if not type(x) == torch.Tensor:
        x = torch.tensor(x)

    x = x.to(device)
    # Suy diễn x
    output = model(x)

    return torch.argmax(output).item()

    
