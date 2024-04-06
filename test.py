import os
import pytz
from tqdm import tqdm
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from dataset.dataset import ButterflyMothDataset
from model.ResNet import ResNet50
from model.VGGNet import VGGNet19


def get_model(args):
    if args.model == "VGGNet19":
        model = VGGNet19()
    elif args.model == "ResNet50":
        model = ResNet50()
    else:
        raise ValueError("Invalid model name")

    return model


def test(args, model, test_dataset):
    # prepare data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # testing
    model.eval()

    test_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = model(inputs)

            preds = outputs.argmax(1)

            test_accuracy += (preds == labels).sum().item()

    test_accuracy = test_accuracy / len(test_loader.dataset)

    return test_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Butterfly & Moth Classification with VGGNet19 / ResNet50"
    )

    parser.add_argument(
        "--data_path", type=str, default="./data", help="path to dataset"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--model", type=str, default="ResNet50", choices=["VGGNet19", "ResNet50"]
    )
    parser.add_argument("--model_path", type=str, default=None, help="Model path")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # load args
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    test_dataset = ButterflyMothDataset(root=args.data_path, mode="test")

    # load model
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)

    print(f"Testing model: {model.__name__}\n\tpath: {args.model_path}")

    # test
    test_accuracy = test(args, model, test_dataset)
    print(f"Test accuracy: {test_accuracy:.4f}")
