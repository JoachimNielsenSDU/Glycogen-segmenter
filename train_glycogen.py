import numpy as np
from tqdm import tqdm

from unet import AttUNet
from datagenerator import ImageSegmentationDataset, ImagePatchBuffer
import torch
import torch.optim as optim
from loss import CategoricalFocalLoss
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a U-Net model with attention gates')
    parser.add_argument('--image_dir', type=str, help='Path to directory containing image files',
                        default='data/glycogen')
    parser.add_argument('--label_dir', type=str, help='Path to directory containing label files',
                        default='data/glycogen')
    parser.add_argument('--segment_names', type=str, nargs='+', required=False, help='List of segment names to extract')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weights', type=str, default='',
                        help='Path to model weights for initialization')
    parser.add_argument('--output_dir', type=str, default='weights_glyco', help='Directory to save model weights')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency to save model weights (epochs)')
    args = parser.parse_args()

    if args.segment_names is None:
        args.segment_names = ["Glycogen", "Background"]

    print(f"Using segment names: {args.segment_names}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load all images into memory
    imagebuffer = ImagePatchBuffer(args.image_dir, args.label_dir, segment_names=args.segment_names, patch_size=1024)

    # Define the model
    filters = [8, 16, 32, 64, 128, 256, 512]

    # Define loss function (cross-entropy)
    criterion = CategoricalFocalLoss(num_classes=len(args.segment_names))

    # Split the dataset into training and validation
    val_split = 0.2
    n = len(imagebuffer)
    indices = np.arange(n)
    split = int(np.floor(val_split * n))
    np.random.shuffle(indices)
    train_index, test_index = indices[split:], indices[:split]

    # Create dataloaders
    train_dataset = ImageSegmentationDataset(imagebuffer, augment=True, indices=train_index)
    validation_dataset = ImageSegmentationDataset(imagebuffer, augment=False, indices=test_index)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    # Create the model
    model = AttUNet(1, 1, filters, final_activation='sigmoid', include_reconstruction=False).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler()

    model.train()

    history = []

    # Train the model
    for epoch in range(args.num_epochs):
        total_loss = 0
        tbar = tqdm(total=len(dataloader))
        model.train()
        for i, (images, labels) in enumerate(dataloader):
            # AMP
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Forward pass
                images, labels = images.to(device), labels.to(device)

                unlabelled = torch.sum(labels, axis=1) < 0
                labels = torch.argmax(labels, axis=1)
                labels[unlabelled] = -1
                labels = labels.unsqueeze(1)

                seg = model.forward(images)

                # seg loss + reconstruction loss
                loss = criterion(seg, labels)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            tbar.update(1)
            tbar.set_postfix_str(f'Loss: {loss.item()}')

            # Dataset doesn't raise StopIteration, so we need to break manually
            if i >= len(dataloader) - 1:
                break

        tbar.close()
        print(f'Epoch {epoch + 1} loss: {total_loss / len(dataloader)}', flush=True)

        # Validation
        model.eval()
        total_val_loss = 0
        tbar = tqdm(total=len(dataloader_val))
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader_val):
                images, labels = images.to(device), labels.to(device)

                unlabelled = torch.sum(labels, axis=1) < 0
                labels = torch.argmax(labels, axis=1)
                labels[unlabelled] = -1
                labels = labels.unsqueeze(1)

                seg = model.forward(images)
                loss = criterion(seg, labels)
                total_val_loss += loss.item()

                # Dataset doesn't raise StopIteration, so we need to break manually
                if i >= len(dataloader_val) - 1:
                    break
                tbar.update(1)
                tbar.set_postfix_str(f'Loss: {loss.item()}')
        tbar.close()

        # plot the last image and label
        import matplotlib.pyplot as plt
        idx = np.random.randint(0, len(images))
        #
        plt.figure(figsize=(10, 5))
        # Pred
        plt.subplot(1, 2, 1)
        plt.imshow(images[idx].cpu().numpy().squeeze(), cmap='gray')
        plt.imshow(seg[idx].cpu().numpy().squeeze(), alpha=0.5, cmap='jet')
        plt.colorbar()

        # GT
        plt.subplot(1, 2, 2)
        plt.imshow(images[idx].cpu().numpy().squeeze(), cmap='gray')
        plt.imshow(labels[idx].cpu().numpy().squeeze(), alpha=0.5, cmap='jet')
        plt.colorbar()

        plt.show()

        print(f'Validation loss: {total_val_loss / len(dataloader_val)}', flush=True)
        history.append({'epoch': epoch, 'train_loss': total_loss / len(dataloader),
                        'val_loss': total_val_loss / len(dataloader_val)})

        if (epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), f'{args.output_dir}/model_epoch_{epoch + 1}.pth')

        print(history)
        # save history
        np.save(f'{args.output_dir}/history.npy', history)
        print('Training complete')


if __name__ == '__main__':
    # add args to argparse
    main()
