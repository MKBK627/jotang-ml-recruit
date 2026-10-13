from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch

X,y = make_moons(
    n_samples = 1000,
    noise = 0.2,
    random_state = 42
)

torch.manual_seed(42)

print("X的形状:",X.shape)
print("y的形状:",y.shape)

plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("make_moons Dataset")
plt.savefig("Jotang-ml-task1/images/make_moons_dataset.png")
plt.show()

from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

print("X_temp:",X_temp.shape)
print("X_test:",X_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size = 0.25,
    random_state = 42,
    stratify = y_temp
)

print("X_train:",X_train.shape)
print("X_val:",X_val.shape)
print("X_test:",X_test.shape)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.title("Training Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("Jotang-ml-task1/images/training_set.png")
plt.show()

plt.scatter(X_val[:,0], X_val[:,1], c=y_val)
plt.title("Validation Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("Jotang-ml-task1/images/validation_set.png")
plt.show()

plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.title("Test Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("Jotang-ml-task1/images/test_set.png")
plt.show()

import torch

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(type(X_train))
print(X_train.dtype)
print(y_train.dtype)

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,hidden_size=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,2)
        )

    def forward(self,x):
        return self.net(x)

model = MLP(hidden_size=16)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = []
train_accuracies = []
val_accuracies = []

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    predicted = torch.argmax(outputs,dim=1)
    train_acc = (predicted == y_train).float().mean()

    with torch.no_grad():
        val_outputs = model(X_val)
        val_predicted = torch.argmax(val_outputs, dim=1)
        val_acc = (val_predicted == y_val).float().mean()

    train_losses.append(loss.item())
    train_accuracies.append(train_acc.item())
    val_accuracies.append(val_acc.item())

    if(epoch + 1) % 50 ==0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.4f}, "
            f"Accuracy: {train_acc.item():.4f}",
            f"Val Acc: {val_acc.item():.4f}"
        )

with torch.no_grad():
    test_outputs = model(X_test)
    test_predicted = torch.argmax(test_outputs,dim=1)
    test_acc = (test_predicted == y_test).float().mean()

print(f"Test Accuracy:{test_acc.item():.4f}")

torch.save(
    model.state_dict(),
    "Jotang-ml-task1/results/mlp_model.pth"
)

loaded_model = MLP()
loaded_model.load_state_dict(
    torch.load("Jotang-ml-task1/results/mlp_model.pth")
)
loaded_model.eval()

print("Model saved and loaded successfully.")

plt.plot(range(1,epochs + 1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("Jotang-ml-task1/images/loss_curve.png")
plt.show()

plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig("Jotang-ml-task1/images/accuracy_curve.png")
plt.show()

import numpy as np

x_min, x_max = X_test[:, 0].min().item() - 0.5, X_test[:, 0].max().item() + 0.5
y_min, y_max = X_test[:, 1].min().item() - 0.5, X_test[:, 1].max().item() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)

with torch.no_grad():
    outputs = model(grid_tensor)
    predictions = torch.argmax(outputs, dim=1)

Z = predictions.numpy().reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(
    X_test[:, 0].numpy(),
    X_test[:, 1].numpy(),
    c=y_test.numpy()
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary")
plt.savefig("Jotang-ml-task1/images/decision_boundary.png")
plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(
    y_test.numpy(),
    test_predicted.numpy()
)

disp = ConfusionMatrixDisplay(
    confusion_matrix = cm,
    display_labels = [0,1]
)

disp.plot()
plt.title("Confusion Matrix")
plt.savefig("Jotang-ml-task1/images/confusion_matrix.png")
plt.show()

print("Confusion Matrix:")
print(cm)

wrong_indices = torch.where(test_predicted != y_test)[0]

print("错误样本数量：",len(wrong_indices))
print("错误样本索引：",wrong_indices)

for idx in wrong_indices:
    print(
        "索引：",idx.item(),
        "真实标签：",y_test[idx].item(),
        "预测标签：",test_predicted[idx].item()
    )

plt.scatter(
    X_test[:, 0].numpy(),
    X_test[:, 1].numpy(),
    c=y_test.numpy()
)

plt.scatter(
    X_test[wrong_indices, 0].numpy(),
    X_test[wrong_indices, 1].numpy(),
    marker="x",
    s=120
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Misclassified Samples")
plt.savefig("Jotang-ml-task1/images/misclassified_samples.png")
plt.show()
