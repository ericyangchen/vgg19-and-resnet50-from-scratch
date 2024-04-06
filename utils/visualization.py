import matplotlib.pyplot as plt


def plot_model_accuracies(data, save_path=None):
    """
    Args:
        data (dict): dictionary containing model names as keys and their accuracies as values
            e.g, data = {
                            "ResNet50_train": [0.1, 0.2, 0.3, 0.4],
                            "ResNet50_valid": [0.1, 0.2, 0.3, 0.4],
                            "VGGNet19_train": [0.2, 0.3, 0.4, 0.5],
                            "VGGNet19_valid": [0.2, 0.3, 0.4, 0.5]
                        }
    """

    for model_name, model_accuracy in data.items():
        plt.plot(model_accuracy, label=model_name)

    plt.title("Accuracy Curve b/w Models")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    if save_path:
        plt.savefig(f"{save_path}/accuracy_curve.png")
