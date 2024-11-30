import numpy as np
import pickle
import os
import cv2

# Function to save the trained MLP model
def save_model(weights_1, biases_1, weights_2, biases_2, filename="mlp_model.pkl"):
    model = {
        'weights_1': weights_1,
        'biases_1': biases_1,
        'weights_2': weights_2,
        'biases_2': biases_2
    }
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

# Function to load the MLP model
def load_model(filename="mlp_model.pkl"):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model['weights_1'], model['biases_1'], model['weights_2'], model['biases_2']

# Load and preprocess images
def load_images_from_directory(base_path, target_size=(28, 28), output_dir="processed"):
    images, labels = [], []
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
    if img.shape[-1] == 4:  # Image has alpha channel
        img = remove_alpha_channel(img)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    block_size = max(3, (min(img.shape) // 10) | 1)  # Ensure odd block size
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2)

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

# Leaky ReLU activation function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward pass of the neural network
def forward_pass(X, weights, biases):
    return np.dot(X, weights) + biases

# Backpropagation to update weights and biases
def backpropagation(X, y, weights_1, biases_1, weights_2, biases_2, hidden_layer_output, output_layer, learning_rate=0.001):
    m = X.shape[0]
    output_error = output_layer - y
    dW_2 = np.dot(hidden_layer_output.T, output_error) / m
    db_2 = np.sum(output_error, axis=0, keepdims=True) / m
    
    hidden_error = np.dot(output_error, weights_2.T) * (hidden_layer_output > 0)
    dW_1 = np.dot(X.T, hidden_error) / m
    db_1 = np.sum(hidden_error, axis=0, keepdims=True) / m
    
    weights_1 -= learning_rate * dW_1
    biases_1 -= learning_rate * db_1
    weights_2 -= learning_rate * dW_2
    biases_2 -= learning_rate * db_2
    
    return weights_1, biases_1, weights_2, biases_2

# Train the MLP model with mini-batch gradient descent
def train_mlp(train_images, train_labels, input_size, hidden_size, output_size, epochs=50, learning_rate=0.001, batch_size=64):
    weights_1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    biases_1 = np.zeros((1, hidden_size))
    
    weights_2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    biases_2 = np.zeros((1, output_size))
    
    y_train = np.eye(output_size)[train_labels]
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        indices = np.random.permutation(len(train_images))
        train_images = train_images[indices]
        y_train = y_train[indices]
        
        for i in range(0, len(train_images), batch_size):
            X_batch = train_images[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            hidden_layer = leaky_relu(forward_pass(X_batch, weights_1, biases_1))
            output_layer = softmax(forward_pass(hidden_layer, weights_2, biases_2))
            
            weights_1, biases_1, weights_2, biases_2 = backpropagation(X_batch, y_batch, weights_1, biases_1, weights_2, biases_2, hidden_layer, output_layer, learning_rate)
        
    return weights_1, biases_1, weights_2, biases_2

# Evaluate model accuracy
def evaluate_model(test_images, test_labels, weights_1, biases_1, weights_2, biases_2):
    hidden_layer = leaky_relu(forward_pass(test_images, weights_1, biases_1))
    output_layer = softmax(forward_pass(hidden_layer, weights_2, biases_2))
    
    predictions = np.argmax(output_layer, axis=1)
    accuracy = np.mean(predictions == test_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    confusion_matrix = np.zeros((10, 10), dtype=np.int32)
    for true, pred in zip(test_labels, predictions):
        confusion_matrix[true, pred] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)

# Main function
def main():
    dataset_path = "data"  # Replace with your dataset's path
    processed_path = "processed"

    print("Loading and preprocessing dataset...")
    images, labels = load_images_from_directory(dataset_path, output_dir=processed_path)
    images = normalize_data(images)
    print(f"Loaded {len(images)} images and saved to {processed_path}/.")

    train_images, test_images, train_labels, test_labels = split_data(images, labels)

    input_size = 28 * 28  # Flattened 28x28 images
    hidden_size = 1024     # Larger hidden layer size for better capacity
    output_size = 10      # Output layer size (10 classes)

    print("Training MLP classifier...")
    weights_1, biases_1, weights_2, biases_2 = train_mlp(train_images, train_labels, input_size, hidden_size, output_size, epochs=50, learning_rate=0.001, batch_size=64)
    print("Training complete.")

    # Save the trained model
    save_model(weights_1, biases_1, weights_2, biases_2)

    print("Testing the model...")
    evaluate_model(test_images, test_labels, weights_1, biases_1, weights_2, biases_2)

if __name__ == "__main__":
    main()
