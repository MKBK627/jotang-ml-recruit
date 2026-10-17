import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,16,kernel_size = 3,padding = 1),  # 输入通道 3，输出通道 16，卷积核 3*3
                                                          # 3 * 128 * 128 -> 16 * 128 * 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),  # 宽高减半 
                                            # 16 * 128 * 128 -> 16 * 64 * 64

            nn.Conv2d(16,32,kernel_size = 3,padding = 1), # 16 * 64 * 64 -> 32 * 64 * 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # 32 * 64 * 64 -> 32 * 32 * 32

            nn.Conv2d(32,64,kernel_size = 3,padding = 1), # 32 * 32 * 32 -> 64 * 32 * 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # 64 * 32 * 32 -> 64 * 16 * 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = SimpleCNN()

model.load_state_dict(
    torch.load("Jotang-ml-task2/results/best_model_aug.pth")
)

model.eval()

image_path = input("请输入要预测的图片路径：").strip('"')
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

probabilities = torch.softmax(output, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()
predicted_prob = probabilities[0, predicted_class].item()

class_names = ["Cat", "Dog"]

print("预测类别：", class_names[predicted_class])
print(f"预测概率：{predicted_prob * 100:.2f}%")

processed_image = input_tensor[0].permute(1, 2, 0).numpy()

plt.imshow(processed_image)
plt.title(
    f"{class_names[predicted_class]} "
    f"{predicted_prob * 100:.2f}%"
)
plt.axis("off")

plt.savefig(
    "Jotang-ml-task2/images/inference_result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
