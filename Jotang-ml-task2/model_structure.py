import matplotlib.pyplot as plt

layers = [
    "Input\n3×128×128",
    "Conv2d\n3→16",
    "ReLU",
    "MaxPool\n16×64×64",
    "Conv2d\n16→32",
    "ReLU",
    "MaxPool\n32×32×32",
    "Conv2d\n32→64",
    "ReLU",
    "MaxPool\n64×16×16",
    "Flatten\n16384",
    "Linear\n16384→128",
    "ReLU",
    "Linear\n128→2",
    "Output\nCat / Dog"
]

plt.figure(figsize=(6, 14))

for i, layer in enumerate(layers):
    y = len(layers) - i

    plt.text(
        0.5,
        y,
        layer,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", alpha=0.3)
    )

    if i < len(layers) - 1:
        plt.arrow(
            0.5,
            y - 0.35,
            0,
            -0.45,
            head_width=0.03,
            head_length=0.12,
            length_includes_head=True
        )

plt.xlim(0, 1)
plt.ylim(0, len(layers) + 1)
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Jotang-ml-task2/images/model_structure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()