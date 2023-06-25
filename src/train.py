import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from model import AE
from input import get_input_loaders

def main():
    train_loader, test_dataset = get_input_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 20
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, 784).to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(train_loader)
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_dataset.data[index].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            test_data = test_dataset.data[index]
            test_data = test_data.to(device)
            test_data = test_data.float()
            test_data = test_data.view(-1, 784)
            output = model(test_data)
            plt.imshow(output.cpu().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

if __name__ == '__main__':
    main()