import os
import shutil
import argparse
import numpy as np
from clustering import Clustering
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image clustering script with output options.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the folder containing all images.")
    parser.add_argument("--n_clusters", type=int, required=True, help="Number of clusters.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory for reorganized images.")
    parser.add_argument("--in_place", action='store_true', help="Reorganize images in place.")
    return parser.parse_args()

def prepare_dataloader(image_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def copy_or_move_images(image_paths, labels, output_dir, in_place):
    if in_place:
        for cluster_id in np.unique(labels):
            cluster_folder = os.path.join(output_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_folder, exist_ok=True)
        
        for img_path, label in zip(image_paths, labels):
            shutil.move(img_path, os.path.join(output_dir, f"cluster_{label}", os.path.basename(img_path)))
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for cluster_id in np.unique(labels):
            cluster_folder = os.path.join(output_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_folder, exist_ok=True)
        
        for img_path, label in zip(image_paths, labels):
            shutil.copy(img_path, os.path.join(output_dir, f"cluster_{label}", os.path.basename(img_path)))

def main():
    args = parse_arguments()

    dataloader = prepare_dataloader(args.image_dir)

    clustering_model = Clustering()
    labels, features = clustering_model.fit(dataloader)
    image_paths = [path for path, _ in dataloader.dataset.samples]

    if args.in_place:
        copy_or_move_images(image_paths, labels, args.image_dir, in_place=True)
    else:
        if not args.output_dir:
            print("Error: Please provide an output directory using --output_dir if not clustering in place.")
            return
        copy_or_move_images(image_paths, labels, args.output_dir, in_place=False)

if __name__ == "__main__":
    main()
