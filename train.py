import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from backbone import ResNet50
from loss import ArcFaceLoss

from utils.dataloaders import create_dataloader
from utils.device import select_device

def train(args):
    num_workers = os.cpu_count() if args.num_workers == -1 else args.num_workers
    
    args.image_size = (args.image_size, args.image_size) if isinstance(args.image_size, int) else args.image_size
    
    train_dir = os.path.join(args.data, 'train')
    train_dataloader, train_datasets = create_dataloader(train_dir, args.image_size, args.batch_size, num_workers)
    
    valid_dir = os.path.join(args.data, 'valid')
    valid_dataloader, _ = create_dataloader(valid_dir, args.image_size, args.batch_size, num_workers)
    
    device = select_device(args.device)
    
    num_classes = len(train_datasets.classes)
    
    feature_extraction = ResNet50(args.embedding_size).to(device)
    criterion = ArcFaceLoss(num_classes, args.embedding_size).to(device)

    # feature_extraction = torch.compile(feature_extraction) # torch 2.0
    # criterion = torch.compile(criterion) # torch 2.0
    
    optimizer = torch.optim.AdamW(
        params=[{'params': feature_extraction.parameters(), 
                 'params': criterion.parameters()}],
        lr=args.learning_rate
    )
    
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
            
            accuracy = torch.sum(logits.argmax(dim=1) == y)
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

                accuracy = torch.sum(logits.argmax(dim=1) == y)
                valid_accuracy += accuracy.item()
                
            valid_loss /= len(valid_dataloader)
            valid_accuracy /= len(valid_dataloader)
        
        print(f'epoch {epoch + 1}/{args.epochs} - loss: {train_loss: .4f} - acc: {train_accuracy: .4f} - val_loss: {valid_loss: .4f} - val_acc: {valid_accuracy: .4f}')
        
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3)
    
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, nargs='+', default=300)
    parser.add_argument('--num-workers', type=int, default=-1)
    
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--margin-loss', type=float, default=0.3)
    parser.add_argument('--scale-loss', type=float, default=30)
    
    parser.add_argument('--device', type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_opt()
    train(args)
    

if __name__ == '__main__':
    main()