from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloader(dir, image_size, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((image_size[0], image_size[1])),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(
        root=dir,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    return dataloader, dataset


if __name__ == '__main__':
    import os
    
    dataloader, dataset = create_dataloaders('data/train', 224, 8, os.cpu_count())
    print(dataloader, dataset)