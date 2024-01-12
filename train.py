# When i started to write this code
# Only 2 guys can understand (I and God)
# Now there's only 1 left (God)

import os
import argparse

import torch

from backbone import ResNet50
from loss import ArcFaceLoss

from utils.dataloader import create_dataloader
from utils.device import select_device


def train(args):
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()
    else:
        args.num_workers

    if isinstance(args.image_size, int):
        args.image_size = (args.image_size, args.image_size)
    else:
        args.image_size

    print(f"Loading trainning data:")
    train_dir = os.path.join(args.data, "train")
    print(f"- Train data path: {train_dir}")
    train_dataloader, train_dataset = create_dataloader(
        train_dir, args.image_size, args.batch_size, args.num_workers
    )
    print(f"- Number of train dataloader: {len(train_dataloader)}")
    num_classes = len(train_dataset.classes)
    print(f"- Number of class: {num_classes}")

    print("Loading validation data:")
    valid_dir = os.path.join(args.data, "valid")
    print(f"- Valid data path: {valid_dir}")
    valid_dataloader, _ = create_dataloader(
        valid_dir, args.image_size, args.batch_size, args.num_workers
    )
    print(f"- Number of valid dataloader: {len(valid_dataloader)}")

    device = select_device(args.device)
    print(f"Using device: {device}")

    feature_extraction = ResNet50(args.embedding_size).to(device)
    criterion = ArcFaceLoss(num_classes, args.embedding_size).to(device)

    # # use with torch version > 2.0 and linux
    # feature_extraction = torch.compile(
    #     feature_extraction
    # )
    # criterion = torch.compile(criterion)

    optimizer = torch.optim.AdamW(
        params=[
            {
                "params": feature_extraction.parameters(),
                "params": criterion.parameters(),
            }
        ],
        lr=args.learning_rate,
    )

    print("Trainning")
    # Trainning model
    train_loss = 0
    train_accuracy = 0
    for epoch in range(args.epochs):
        # Training step
        feature_extraction.train()
        criterion.train()

        for X, y in train_dataloader:
            optimizer.zero_grad()

            X = X.to(device)
            y = y.to(device)

            embeddings = feature_extraction(X)
            logits, loss = criterion(embeddings, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            accuracy = torch.sum(logits.argmax(dim=1) == y) / len(y)
            train_accuracy += accuracy.item()

        train_loss /= len(train_dataloader)
        train_accuracy /= len(train_dataloader)

        # Validation step
        feature_extraction.eval()
        criterion.eval()
        valid_loss = 0
        valid_accuracy = 0
        with torch.no_grad():
            for X, y in valid_dataloader:
                X = X.to(device)
                y = y.to(device)

                embeddings = feature_extraction(X)
                logits, loss = criterion(embeddings, y)

                valid_loss += loss.item()

                accuracy = torch.sum(logits.argmax(dim=1) == y) / len(y)
                valid_accuracy += accuracy.item()

            valid_loss /= len(valid_dataloader)
            valid_accuracy /= len(valid_dataloader)

        print(
            f"- Epoch {epoch + 1}/{args.epochs} - loss: {train_loss: .4f} - acc: {train_accuracy: .4f} - val_loss: {valid_loss: .4f} - val_acc: {valid_accuracy: .4f}"
        )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-3)

    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, nargs="+", default=224)

    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--margin-loss", type=float, default=0.3)
    parser.add_argument("--scale-loss", type=float, default=30)

    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_opt()
    train(args)


if __name__ == "__main__":
    main()
