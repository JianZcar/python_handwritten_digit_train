import cv2
import numpy as np
import os

def load_images_from_directory(base_path, target_size=(28, 28), output_dir="processed"):
    images = []
    labels = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in range(10):  # Assuming folders are named '0' to '9'
        folder_path = os.path.join(base_path, str(label))
        output_folder = os.path.join(output_dir, str(label))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} does not exist. Skipping.")
            continue
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load image with alpha channel if present
            if img is None:
                print(f"Warning: Unable to load {img_path}. Skipping.")
                continue
            
            # If image has transparency (alpha channel), replace it with white
            if len(img.shape) == 3 and img.shape[2] == 4:  # Check for alpha channel
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Remove alpha channel
                img_rgb[img[:, :, 3] == 0] = [255, 255, 255]  # Set transparent pixels to white
            else:
                img_rgb = img  # No transparency, just use the image as is
            
            # If the image has 3 channels (RGB/BGR), convert to grayscale
            if len(img_rgb.shape) == 3:
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_rgb  # Already grayscale (1 channel)

            # Apply adaptive thresholding with dynamic block size
            block_size = max(3, (min(img_gray.shape) // 10) | 1)  # Ensure odd block size
            img_bin = cv2.adaptiveThreshold(
                img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2
            )

            # Resize the image with aspect ratio maintenance
            processed_img = resize_with_aspect_ratio(img_bin, target_size)
            
            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_img)
            
            # Flatten the image to a 1D array for training
            images.append(processed_img.flatten())
            labels.append(label)
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

def resize_with_aspect_ratio(image, target_size=(28, 28)):
    """
    Resize an image while maintaining its aspect ratio. The resized image is placed on a white canvas of the target size.
    """
    h, w = image.shape

    # Calculate scaling factor
    scale = min(target_size[0] / h, target_size[1] / w)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create a white canvas
    canvas = np.full(target_size, 255, dtype=np.uint8)

    # Center the resized image on the canvas
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return canvas

# Normalize the dataset
def normalize_data(images):
    return images / 255.0

# Stratified dataset splitting
def split_data(images, labels, test_size=0.2):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    test_indices = []
    train_indices = []

    for label, count in zip(unique_labels, label_counts):
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        split_idx = int(count * (1 - test_size))
        train_indices.extend(label_indices[:split_idx])
        test_indices.extend(label_indices[split_idx:])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    train_images, test_images = images[train_indices], images[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]
    
    return train_images, test_images, train_labels, test_labels

def evaluate_model(results, true_labels):
    from collections import Counter
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

    train_images, test_images, train_labels, test_labels = split_data(images, labels, test_size=0.2)

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
    ret, results, neighbors, dist = knn.findNearest(test_images, k=5)

    evaluate_model(results, test_labels)

if __name__ == "__main__":
    main()
