import os
import argparse

from utils.extract_tensorboard_data import (
    extract_training_logs,
    get_values_from_events,
)
from utils.visualization import plot_model_accuracies


def compare_models(args):
    model_paths = args.model_paths

    model_names = []
    identify_models_with_id = False

    data = []
    for model_path in model_paths:
        paths = model_path.strip().strip("/").split("/")
        model_name, model_id = paths[-2], paths[-1]

        # check for duplicated model names (ResNet50 or VGGNet19)
        if model_name in model_names:
            identify_models_with_id = True
        elif not identify_models_with_id:
            model_names.append(model_name)

        # transform logs
        logs = extract_training_logs(os.path.join(model_path, "logs"))
        logs["model_name"] = model_name
        logs["model_id"] = model_id
        logs["Train/Accuracy"] = get_values_from_events(logs["Train/Accuracy"])
        logs["Validation/Accuracy"] = get_values_from_events(
            logs["Validation/Accuracy"]
        )
        logs["Train/Loss"] = get_values_from_events(logs["Train/Loss"])
        logs["highest_train_accuracy"] = max(logs["Train/Accuracy"])

        data.append(logs)

    # plot accuracy
    plot_data = dict()
    for model in data:
        # set legends
        if identify_models_with_id:
            legend_train = f"{model['model_name']} ({model['model_id']}) - train"
            legend_valid = f"{model['model_name']} ({model['model_id']}) - valid"
        else:
            legend_train = f"{model['model_name']} - train"
            legend_valid = f"{model['model_name']} - valid"

        plot_data[legend_train] = model["Train/Accuracy"]
        plot_data[legend_valid] = model["Validation/Accuracy"]

    plot_model_accuracies(data=plot_data, save_path=args.output_dir)

    # print model info
    seperator = "----------------------------------"
    print(seperator)
    for model in data:
        if identify_models_with_id:
            print(f"Model: {model['model_name']} ({model['model_id']})")
        else:
            print(f"Model: {model['model_name']}")

        print(
            f"Train Accuracy: {model['highest_train_accuracy'] * 100:.2f}%, Train Loss: {model['Train/Loss'][-1]:.4f}"
        )
        print(seperator)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare models")

    parser.add_argument("--data", type=str, default="./data", help="Path to dataset")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        help="Paths to model directories",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/comparisons",
        help="Path to comparison output",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # compare models
    compare_models(args)
