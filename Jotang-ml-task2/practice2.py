import os 
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class CatsDogsDataset(Dataset):
    def __init__(self,data_dir,transform = None):
        self.data_dir = data_dir
        self.transform = transform

        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".jpg",".jpeg",".png"))
            and(
                f.lower().startswith("cat.")
                or f.lower().startswith("dog.")
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        file_name = self.files[index]

        file_path = os.path.join(
            self.data_dir,
            file_name
        )

        image = Image.open(file_path).convert("RGB")

        if file_name.lower().startswith("cat."):
            label = 0
        else:
            label = 1

        if self.transform:
            image = self.transform(image) 

        return image,label   

data_dir =  input("请输入 Kaggle Dogs vs. Cats 训练集文件夹路径：").strip('"')

all_files = [
    f for f in os.listdir(data_dir)
    if f.lower().endswith((".jpg",".jpeg",".png"))
]

cat_files = [
    f for f in all_files
    if f.lower().startswith("cat.")
]
dog_files = [
    f for f in all_files
    if f.lower().startswith("dog.")
]

print("小猫的图片数量：", len(cat_files))
print("小狗的图片数量：", len(dog_files))


sample_cats = random.sample(cat_files,3)
sample_dogs = random.sample(dog_files,3)

sample_files = [
    (os.path.join(data_dir,f),"Cat") 
    for f in sample_cats
] + [
    (os.path.join(data_dir,f),"Dog") 
    for f in sample_dogs
]

plt.figure(figsize=(12,8))

for i,(file_path,label) in enumerate(sample_files):
    image = Image.open(file_path)

    print(
        label,
        os.path.basename(file_path),
        "size:",
        image.size
    )

    plt.subplot(2,3,i+1)
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")

plt.tight_layout()
plt.show()

train_transform  = transforms.Compose([
    transforms.Resize((144,144)),
    transforms.RandomCrop((128,128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

no_aug_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

eval_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# 随机选择一张图片，展示预处理前后效果
sample_file = random.choice(all_files)
sample_path = os.path.join(data_dir, sample_file)

original_image = Image.open(sample_path).convert("RGB")

# 使用验证/测试集的固定预处理
processed_image = eval_transform(original_image)
processed_image_show = processed_image.permute(1, 2, 0).numpy()

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Before Preprocessing")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed_image_show)
plt.title("After Preprocessing")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    "Jotang-ml-task2/images/preprocessing_compare.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# 展示同一张图片多次随机数据增强后的效果
plt.figure(figsize=(12, 3))

plt.subplot(1, 4, 1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")

for i in range(3):
    augmented_image = train_transform(original_image)

    augmented_image = (
        augmented_image
        .permute(1, 2, 0)
        .numpy()
    )

    plt.subplot(1, 4, i + 2)
    plt.imshow(augmented_image)
    plt.title(f"Augmented {i + 1}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    "Jotang-ml-task2/images/augmentation_samples.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

base_dataset = CatsDogsDataset(
    data_dir = data_dir,
    transform = None
)

print("总图片数量：", len(base_dataset))

labels = [
    0 if file_name.lower().startswith("cat.") else 1
    for file_name in base_dataset.files
]

indices = list(range(len(base_dataset)))

train_val_indices,test_indices = train_test_split(
    indices,
    test_size = 0.2,
    random_state = 42,
    stratify = labels
)

train_val_labels = [
    labels[i] 
    for i in train_val_indices
]

train_indices, val_indices = train_test_split(
    train_val_indices,
    test_size=0.25,
    random_state=42,
    stratify=train_val_labels
)

train_dataset_full = CatsDogsDataset(
    data_dir = data_dir,
    transform=train_transform
)

eval_dataset_full = CatsDogsDataset(
    data_dir = data_dir,
    transform=eval_transform
)

train_dataset_aug = Subset(
    CatsDogsDataset(
        data_dir=data_dir,
        transform=train_transform
    ),
    train_indices
)  # 训练集

train_dataset_no_aug = Subset(
    CatsDogsDataset(
        data_dir=data_dir,
        transform=no_aug_transform
    ),
    train_indices
)  # 训练集

val_dataset = Subset(eval_dataset_full, val_indices)  # 验证集
test_dataset = Subset(eval_dataset_full, test_indices)  # 测试集

print("有增强的训练集数量：", len(train_dataset_aug))
print("无增强的训练集数量：", len(train_dataset_no_aug))
print("验证集数量：", len(val_dataset))
print("测试集数量：", len(test_dataset))

train_loader_aug = DataLoader(
    train_dataset_aug,
    batch_size=16,
    shuffle=True
)  # 训练集

train_loader_no_aug = DataLoader(
    train_dataset_no_aug,
    batch_size=16,
    shuffle=True
)  # 训练集

val_loader = DataLoader(
    val_dataset,
    batch_size = 16,
    shuffle = False
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 16,
    shuffle = False
)

images, labels = next(iter(train_loader_aug))

print("images shape:",images.shape)
print("labels shape:",labels.shape)


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

# 封装train_model
def train_model(model,train_loader,val_loader,save_path,epochs=3):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 0.001
    )

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images,labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            predicted = torch.argmax(outputs,dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval() # 验证/测试阶段

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)

                predicted = torch.argmax(outputs, dim=1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                save_path
            )

            print("保存当前最佳模型")

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

    return model,train_losses,train_accuracies,val_accuracies,best_val_acc


torch.manual_seed(42)
model_aug = SimpleCNN()

model_aug, train_losses_aug, train_acc_aug, val_acc_aug, best_val_aug = train_model(
    model_aug,
    train_loader_aug,
    val_loader,
    "Jotang-ml-task2/results/best_model_aug.pth"
)

torch.manual_seed(42)
model_no_aug = SimpleCNN()

model_no_aug, train_losses_no_aug, train_acc_no_aug, val_acc_no_aug, best_val_no_aug = train_model(
    model_no_aug,
    train_loader_no_aug,
    val_loader,
    "Jotang-ml-task2/results/best_model_no_aug.pth"
)

# 绘制 Loss 曲线
epochs_range = range(1, len(train_losses_aug) + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_losses_aug,
    marker="o",
    label="With Augmentation"
)

plt.plot(
    epochs_range,
    train_losses_no_aug,
    marker="o",
    label="Without Augmentation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "Jotang-ml-task2/images/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# 绘制 Accuracy 曲线
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_acc_aug,
    marker="o",
    label="Train Acc - Aug"
)

plt.plot(
    epochs_range,
    val_acc_aug,
    marker="o",
    label="Val Acc - Aug"
)

plt.plot(
    epochs_range,
    train_acc_no_aug,
    marker="o",
    label="Train Acc - No Aug"
)

plt.plot(
    epochs_range,
    val_acc_no_aug,
    marker="o",
    label="Val Acc - No Aug"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "Jotang-ml-task2/images/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

best_model_aug = SimpleCNN()
best_model_aug.load_state_dict(
    torch.load("Jotang-ml-task2/results/best_model_aug.pth")
)
best_model_aug.eval()

best_model_no_aug = SimpleCNN()
best_model_no_aug.load_state_dict(
    torch.load("Jotang-ml-task2/results/best_model_no_aug.pth")
)
best_model_no_aug.eval()


def test_model(model, test_loader):
    test_correct = 0
    test_total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            predicted = torch.argmax(outputs, dim=1)

            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total

    return test_acc

test_acc_aug = test_model(
    best_model_aug,
    test_loader
)

test_acc_no_aug = test_model(
    best_model_no_aug,
    test_loader
)

print("有增强的Test Accuracy:", test_acc_aug)
print("无增强的Test Accuracy:", test_acc_no_aug)

if best_val_aug >= best_val_no_aug:
    best_model = best_model_aug
    print("后续使用有增强的模型进行可视化")
else:
    best_model = best_model_no_aug
    print("后续使用无增强的模型进行可视化")


images, labels = next(iter(train_loader_aug))

sample_image = images[0].unsqueeze(0)

with torch.no_grad():
    outputs = best_model(images)

    print("输入shape:", images.shape)
    print("输出shape:", outputs.shape)

    x = sample_image

    print("输入：", x.shape)

    for layer in best_model.features:
        x = layer(x)
        print(layer.__class__.__name__, "->", x.shape)

    for layer in best_model.classifier:
        x = layer(x)
        print(layer.__class__.__name__, "->", x.shape)

# 从测试集中选择 3 张图片观察第一层卷积特征
sample_indices = [0, 1, 2]

plt.figure(figsize=(12, 9))

for row, index in enumerate(sample_indices):
    sample_image, sample_label = test_dataset[index]
    input_image = sample_image.unsqueeze(0)
    with torch.no_grad():
        first_conv_output = best_model.features[0](input_image)

    # 第一列显示原图
    plt.subplot(3, 5, row * 5 + 1)
    plt.imshow(
        sample_image.permute(1, 2, 0).numpy()
    )
    label_name = "Cat" if sample_label == 0 else "Dog"
    plt.title(f"Original\n{label_name}")
    plt.axis("off")

    # 后四列显示不同通道的特征图
    for channel in range(4):
        plt.subplot(
            3,
            5,
            row * 5 + channel + 2
        )
        feature_map = (
            first_conv_output[0, channel]
            .numpy()
        )
        plt.imshow(
            feature_map,
            cmap="gray"
        )
        plt.title(f"Channel {channel}")
        plt.axis("off")

print(
    "第一层卷积输出shape:",
    first_conv_output.shape
)

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/first_feature_maps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 从测试集中选择 3 张图片观察最后一层卷积特征
plt.figure(figsize=(12, 9))

for row, index in enumerate(sample_indices):

    sample_image, sample_label = test_dataset[index]

    input_image = sample_image.unsqueeze(0)

    x = input_image

    with torch.no_grad():
        # 运行到最后一个卷积层
        for layer in best_model.features[:7]:
            x = layer(x)

    last_conv_output = x

    # 第一列显示原图
    plt.subplot(3, 5, row * 5 + 1)

    plt.imshow(
        sample_image.permute(1, 2, 0).numpy()
    )

    label_name = "Cat" if sample_label == 0 else "Dog"

    plt.title(f"Original\n{label_name}")
    plt.axis("off")

    # 后四列显示特征图
    for channel in range(4):

        plt.subplot(
            3,
            5,
            row * 5 + channel + 2
        )

        feature_map = (
            last_conv_output[0, channel]
            .numpy()
        )

        plt.imshow(
            feature_map,
            cmap="gray"
        )

        plt.title(f"Channel {channel}")
        plt.axis("off")

print(
    "最后一层卷积输出shape:",
    last_conv_output.shape
)

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/last_feature_maps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

wrong_images = []
wrong_labels = []
wrong_preds = []

best_model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        outputs = best_model(images)
        predicted = torch.argmax(outputs, dim=1)

        wrong_mask = predicted != labels

        wrong_images.extend(images[wrong_mask])
        wrong_labels.extend(labels[wrong_mask])
        wrong_preds.extend(predicted[wrong_mask])

print("错误样本数量：", len(wrong_images))

plt.figure(figsize=(12, 8))

show_num = min(6, len(wrong_images))

for i in range(show_num):
    image = wrong_images[i].permute(1, 2, 0).numpy()

    true_label = "Cat" if wrong_labels[i].item() == 0 else "Dog"
    pred_label = "Cat" if wrong_preds[i].item() == 0 else "Dog"

    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/misclassified_samples.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()