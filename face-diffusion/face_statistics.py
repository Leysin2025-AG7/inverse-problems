import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image

N_PIX = 256
EYE_R = 122
EYE_C = 95


class FlatImageDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images from a flat directory structure.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Supported image extensions
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        # Get all image files
        self.image_paths = []
        if os.path.exists(root_dir):
            for filename in os.listdir(root_dir):
                if any(filename.lower().endswith(ext) for ext in self.image_extensions):
                    self.image_paths.append(os.path.join(root_dir, filename))

        self.image_paths.sort()  # Sort for reproducibility

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = None

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return image and dummy label (0) since we don't need labels for statistics
        return image, 0


def load_and_preprocess_faces(data_root="./data", batch_size=64, num_workers=4):
    """
    Load face dataset from local directory and apply preprocessing transformations.
    Note: The face dataset was downloaded manually from
    https://github.com/ziqihuangg/CelebA-Dialog
    into the folder `./data/images/` before running this script.

    Args:
        data_root (str): Root directory containing the face images
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes for data loading

    Returns:
        DataLoader: DataLoader for the preprocessed face dataset
    """
    # Check if data directory exists
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory {data_root} not found!")

    # Define preprocessing transformations
    transform = transforms.Compose(
        [
            transforms.Resize((N_PIX, N_PIX)),  # Resize to N_PIXxN_PIX pixels
            transforms.ToTensor(),  # Convert PIL Image with uint8 RGB values to tensor with values in [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Rescale to [-1,1]
        ]
    )

    # Try to load as ImageFolder first (expects subdirectories)
    try:
        dataset = ImageFolder(root=data_root, transform=transform)
        print("Loaded dataset using ImageFolder structure")
    except RuntimeError:
        # If ImageFolder fails, create a custom dataset for flat directory structure
        print("ImageFolder failed, trying flat directory structure...")
        dataset = FlatImageDataset(data_root, transform=transform)

    if len(dataset) == 0:
        raise ValueError(
            f"No images found in {data_root}. Please ensure images are placed in the directory."
        )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for statistics computation
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader, len(dataset)


class RunningMoments:
    """
    Class for computing mean and variance of an image dataset in multiple batches
    """

    def __init__(self, im_shape):
        self.sum_accumulator = torch.zeros(im_shape)
        self.sum_squares_accumulator = torch.zeros(im_shape)
        self.num_images = 0

    def add_batch(self, images):
        """
        Update the mean and variance with a new batch of images

        Arguments:
            images - tensor of shape BCHW
        """
        # images shape: (batch_size, 3, N_PIX, N_PIX)
        self.num_images += images.shape[0]

        # update sum and sum of squares for mean and variance computation
        self.sum_accumulator += torch.sum(images, dim=0)
        self.sum_squares_accumulator += torch.sum(images**2, dim=0)

    @property
    def mean(self):
        """Compute mean: E(X)"""
        return self.sum_accumulator / self.num_images if self.num_images > 0 else None

    @property
    def variance(self):
        """Compute sample variance: Var(X) = E[X**2] - E[X]**2"""
        mean_of_squares = self.sum_squares_accumulator / (self.num_images - 1)
        if self.num_images == 0:
            return None
        elif self.num_images == 1:
            return torch.zeros_like(mean_of_squares)
        else:
            assert self.mean is not None
            return (self.sum_squares_accumulator - self.num_images * self.mean**2) / (
                self.num_images - 1
            )

    def print_statistics_summary(self):
        """
        Print summary statistics of the computed mean and variance.
        """
        print("\n" + "=" * 50)
        print("DATASET STATISTICS SUMMARY")
        print("=" * 50)

        mean = self.mean
        variance = self.variance
        assert mean is not None and variance is not None, "No moments computed yet!"

        # Overall statistics
        print(f"Mean tensor shape: {mean.shape}")
        print(f"Variance tensor shape: {variance.shape}")

        # Channel-wise statistics
        for channel in range(3):
            channel_name = ["Red", "Green", "Blue"][channel]
            print(f"\n{channel_name} Channel:")
            print(
                f"  Mean range: [{mean[channel].min():.4f}, {mean[channel].max():.4f}]"
            )
            print(f"  Mean average: {mean[channel].mean():.4f}")
            print(
                f"  Variance range: [{variance[channel].min():.4f}, {variance[channel].max():.4f}]"
            )
            print(f"  Variance average: {variance[channel].mean():.4f}")
            print(
                f"  Standard deviation average: {torch.sqrt(variance[channel]).mean():.4f}"
            )

    def save_statistics(self, save_path="./data"):
        """
        Save mean and variance tensors as NPZ files for exact preservation.

        Args:
            save_path (str): Directory to save the NPZ files
        """

        mean = self.mean
        variance = self.variance
        assert mean is not None and variance is not None, "No moments computed yet!"

        os.makedirs(save_path, exist_ok=True)

        # Convert tensors to numpy arrays
        mean_np = mean.numpy()
        variance_np = variance.numpy()

        # Save mean as NPZ
        mean_path = os.path.join(save_path, f"mean{N_PIX}.npz")
        np.savez(mean_path, mean=mean_np)

        # Save variance as NPZ
        variance_path = os.path.join(save_path, f"variance{N_PIX}.npz")
        np.savez(variance_path, variance=variance_np)

        print("\nStatistics images saved as NPZ files:")
        print(f"  - Mean: {mean_path}")
        print(f"  - Variance: {variance_path}")
        print(
            "  - To load again: np.load('mean.npz')['mean'] or np.load('variance.npz')['variance']"
        )


class SinglePixelVals:
    """Simple class to append the values of a single pixel across multiple batches"""

    def __init__(self, shape, pixel, dataset_size):
        assert len(shape) in [2, 3], "Got invalid image shape"
        assert 0 <= pixel[0] and pixel[0] < shape[1], (
            "Selected pixel is outside vertical image range"
        )
        assert 0 <= pixel[1] and pixel[1] < shape[2], (
            "Selected pixel is outside horizontal image range"
        )

        num_channels = shape[0]
        self.pixel_r = pixel[0]
        self.pixel_c = pixel[1]
        self.pixel_vals = torch.zeros((dataset_size, num_channels))
        self.idx = 0

    def add_values(self, images):
        """Add values of an image batch"""
        batch_size = images.shape[0]
        self.pixel_vals[self.idx : self.idx + batch_size, :] = images[
            :, :, self.pixel_r, self.pixel_c
        ]
        self.idx += batch_size

    def save_values(self, save_path):
        """Saves pixel values as npz file under given path"""
        os.makedirs(save_path, exist_ok=True)

        # Convert tensor to numpy array
        pixel_vals_np = self.pixel_vals[: self.idx, :].numpy()
        # convert to uint8 [0, 255]
        pixel_vals_rgb = ((pixel_vals_np * 0.5 + 0.5) * 255.0).astype(np.uint8)

        # Save values as NPZ
        pixels_path = os.path.join(save_path, f"pixel_values{N_PIX}.npz")
        np.savez(pixels_path, pixel_values=pixel_vals_rgb)

        print(f"\nSingle pixel values saved as uint8 RGB in NPZ file {pixels_path}")
        print(f"  - To load again: np.load('{pixels_path}')['pixel_values']")


def main():
    """
    Main function to run the experiment.
    """
    print("Face Dataset Statistics Computation")
    print("=" * 50)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess dataset
    print("\nLoading face dataset...")
    dataloader, dataset_size = load_and_preprocess_faces(
        data_root="./data", batch_size=64, num_workers=4
    )

    print("Dataset loaded successfully!")
    print(f"Total images: {dataset_size}")
    print(f"Image shape after preprocessing: (3, {N_PIX}, {N_PIX})")
    print("Pixel value range: [-1, 1]")

    images_moments = RunningMoments((3, N_PIX, N_PIX))
    eye_pixels = SinglePixelVals((3, N_PIX, N_PIX), (EYE_R, EYE_C), dataset_size)
    print(
        "Batchwise computation of empirical mean and variance and collection of eye pixels..."
    )
    for batch_idx, (images, _) in enumerate(dataloader):
        # images shape: (batch_size, 3, N_PIX, N_PIX)
        batch_size = images.size(0)

        # update running sums for mean and variance computation
        images_moments.add_batch(images)

        # extract values of single pixel in eye
        eye_pixels.add_values(images)

        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {(batch_idx + 1) * batch_size} images")

    # Print summary
    images_moments.print_statistics_summary()

    # Save statistics as NPZ images
    images_moments.save_statistics(save_path="./statistics")

    # Save the eye pixel values as NPZ file
    eye_pixels.save_values(save_path="./eye_pixels")

    print("\nStatistics computed successfully!")

    return


if __name__ == "__main__":
    main()
