import cv2
import numpy as np
import os

# Load and preprocess images
def load_images_from_directory(base_path, target_size=(28, 28), output_dir="processed"):
    images, labels = [], []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for label in range(10):  # Assuming subfolders are named '0' to '9'
        folder_path = os.path.join(base_path, str(label))
        output_folder = os.path.join(output_dir, str(label))
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} does not exist. Skipping.")
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Unable to load {img_path}. Skipping.")
                continue

            processed_img = preprocess_image(img, target_size)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_img)

            images.append(processed_img.flatten())
            labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

# Preprocess individual images
def preprocess_image(img, target_size):
    # Handle transparency
    if img.shape[-1] == 4:  # Image has alpha channel
        img = remove_alpha_channel(img)

    # Convert to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    block_size = max(3, (min(img.shape) // 10) | 1)  # Ensure odd block size
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2)

    # Resize while maintaining aspect ratio
    return resize_with_aspect_ratio(img_bin, target_size)

# Remove alpha channel and replace transparent pixels with white
def remove_alpha_channel(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_rgb[img[:, :, 3] == 0] = [255, 255, 255]
    return img_rgb

# Resize an image with aspect ratio maintenance
def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h))
    canvas = np.full(target_size, 255, dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return canvas

# Normalize images to [0, 1]
def normalize_data(images):
    return images / 255.0

# Split data into train and test sets
def split_data(images, labels, test_size=0.2):
    np.random.seed(42)  # Ensures reproducibility
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    return images[train_indices], images[test_indices], labels[train_indices], labels[test_indices]

# Evaluate model accuracy
def evaluate_model(results, true_labels):
    accuracy = np.mean(results.flatten() == true_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Confusion matrix
    unique_labels = np.unique(true_labels)
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=np.int32)
    for true, pred in zip(true_labels, results.flatten()):
        confusion_matrix[int(true), int(pred)] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)
    return accuracy

# Main program
def main():
    dataset_path = "data"  # Replace with your dataset's path
    processed_path = "processed"

    print("Loading and preprocessing dataset...")
    images, labels = load_images_from_directory(dataset_path, output_dir=processed_path)
    images = normalize_data(images)
    print(f"Loaded {len(images)} images and saved to {processed_path}/.")

    train_images, test_images, train_labels, test_labels = split_data(images, labels)

    print("Training k-NN classifier...")
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(5)
    knn.setAlgorithmType(cv2.ml.KNearest_BRUTE_FORCE)

    knn.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)
    print("Training complete.")

    model_filename = "knn_model.xml"
    knn.save(model_filename)
    print(f"Model saved to {model_filename}")

    print("Testing the model...")
    _, results, _, _ = knn.findNearest(test_images, k=5)
    evaluate_model(results, test_labels)

if __name__ == "__main__":
    main()
