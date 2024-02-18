import argparse
import pandas as pd
import numpy as np
from keras.datasets import mnist
from sklearn.metrics.pairwise import cosine_distances
import pickle


def load_dataset(mnist_path):
    print('Loading MNIST dataset...')
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print('Loading Complete!')
    # Flatten the images for simplicity in handling
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    print(f"Train images shape: {X_train.shape}, Test images shape: {X_test.shape}")
    return X_train, Y_train, X_test, Y_test


# Specify the paths to your CSV files here
train_file_path = 'MNIST-csv/train.csv'
test_file_path = 'MNIST-csv/test.csv'

#(train_images, train_labels), (test_images) = load_and_prep_data(train_file_path, test_file_path)


def create_data_representation(dim=10000, representation_type='bipolar', len_features=256):
    print("Creating data representation...")
    """
    Generates lookup tables for encoding features into hyperdimensional vectors.

    Args:
    - dim (int): The dimensionality of the hyperdimensional vectors. Default is 10,000.
    - representation_type (str): The type of data representation to use. Currently supports 'binary'.
    - len_features (int): The number of unique features for which to generate lookup tables. Default is 2.

    Returns:
    - tuple of np.ndarray: Two lookup tables, each as a numpy array. The first table for the first feature, 
      and the second table for the second feature. Each table contains randomly generated bipolar (-1 or +1) 
      vectors of length `dim`.
    """
    if representation_type == 'bipolar':
        # Generate a binary representation table where values are either -1 or +1,
        # suitable for hyperdimensional computing contexts.
        representation_table = np.random.randint(2, size=(len_features, dim), dtype=np.int8) * 2 - 1

    return representation_table


def encode_data(data_item, position_table, feature_table, dim=10000):
    #print("Encoding data item...")
    """
    Encodes a single data item (e.g., an image) into a hyperdimensional vector using predefined lookup tables.

    Args:
    - data_item (array-like): The data item to encode, represented as an array of feature values.
    - position_table (np.ndarray): The lookup table for position vectors.
    - feature_table (np.ndarray): The lookup table for feature vectors (e.g., grayscale values).
    - dim (int): The dimensionality of the hyperdimensional space. Default is 10,000.

    Returns:
    - np.ndarray: The encoded hyperdimensional vector representing the input data item.
    """
    item_repr = np.zeros(dim, dtype=np.int16)  # Initialize the item's representation vector.

    # Iterate over each pixel/feature in the data item.
    for pixel in range(len(data_item)):
        # Multiply the position vector by the feature vector (element-wise) to get the pixel's representation.
        pixel_repr = np.multiply(position_table[pixel % len(position_table)], feature_table[int(data_item[pixel]) % len(feature_table)])
        # Add the pixel's representation to the item's overall representation vector.
        item_repr = np.add(item_repr, pixel_repr)

    return item_repr


def train_model(associative_memory, X_train, Y_train, position_table, feature_table, dim=10000):
    """
    Trains the associative memory model using encoded data items and their labels.

    Args:
    - associative_memory (np.ndarray): The initial associative memory to train.
    - X_train (list of array-like): The training data items, each to be encoded.
    - Y_train (list or array-like): The labels corresponding to each training data item.
    - position_table (np.ndarray): The lookup table for position vectors.
    - feature_table (np.ndarray): The lookup table for feature vectors.
    - dim (int): The dimensionality of the hyperdimensional vectors. Default is 10,000.

    Returns:
    - np.ndarray: The trained associative memory, updated based on the training data.
    """
    associative_memory_trained = associative_memory.copy()  # Create a copy of the associative memory to train.
    print("Training model...")
    # Iterate over each data item and its label in the training set.
    for i, (data_x, data_y) in enumerate(zip(X_train, Y_train)):
        # Encode the data item into a hyperdimensional vector.
        encoded_vector = encode_data(data_x, position_table, feature_table, dim)
        # Update the associative memory for the corresponding label with the encoded vector.
        associative_memory_trained[data_y] = np.add(associative_memory_trained[data_y], encoded_vector)
        if i % 10000 == 0:
            print(f"Processed {i} training items")
    return associative_memory_trained


def collapse_probability(associative_memory, data_item, position_table, feature_table, dim=10000):
    #print("Collapsing probability...")
    # Encode the data item into a hyperdimensional vector.
    encoded_item = encode_data(data_item, position_table, feature_table, dim)
    
    # Initialize match to 0 and distance_lowest to the highest possible starting value.
    match = 0
    distance_lowest = 2  # Max possible value for cosine distance in this context.
    
    # Iterate through each class vector in the associative memory to find the closest match.
    for index, class_vector in enumerate(associative_memory):
        # Calculate the cosine distance between the encoded item and the current class vector.
        distance = cosine_distances([encoded_item, class_vector])[0][1]
        
        # Update match if the current distance is lower than the lowest distance found so far.
        if distance < distance_lowest:
            match = index
            distance_lowest = distance
            
    return match, encoded_item
    
def test_data(associative_memory, X_test, Y_test, position_table, feature_table, dim=10000):
    print("Testing model...")
    """
    Tests the trained associative memory model on test data to evaluate its accuracy.

    Args:
    - associative_memory (np.ndarray): The trained associative memory.
    - X_test (np.ndarray): Test images, each encoded as a flattened array.
    - Y_test (np.ndarray): True labels for the test images.
    - position_table (np.ndarray): Lookup table for position vectors.
    - feature_table (np.ndarray): Lookup table for feature vectors (e.g., grayscale values).
    - dim (int): The dimensionality of the hyperdimensional space. Default is 10,000.

    Returns:
    - float: The accuracy of the model on the test data, calculated as the proportion
             of correctly predicted labels.
    """
    correct_predictions = 0
    for i, (data_item, true_label) in enumerate(zip(X_test, Y_test)):
        predicted_label, _ = collapse_probability(associative_memory, data_item, position_table, feature_table, dim)
        if predicted_label == true_label:
            correct_predictions += 1
        if i % 5000 == 0:
            print(f"Tested {i} items succesfully")
    accuracy = correct_predictions / len(X_test)
    print(f"Test accuracy: {accuracy}")
    return accuracy

def retrain_model(associative_memory, X_retrain, Y_retrain, position_table, feature_table, dim=10000):
    """
    Retrains the associative memory model using additional or corrective training data.
    
    Args:
    - associative_memory (np.ndarray): The current associative memory to be updated.
    - X_retrain (np.ndarray): Array of images for retraining, each as a flattened array.
    - Y_retrain (np.ndarray): Array of labels corresponding to the retraining images.
    - position_table (np.ndarray): Lookup table for position vectors used in encoding.
    - feature_table (np.ndarray): Lookup table for feature vectors used in encoding.
    - dim (int): Dimensionality of the hyperdimensional vectors.

    Returns:
    - np.ndarray: Updated associative memory after retraining with additional data.
    """
    am_ = associative_memory.copy()  # Make a copy to avoid modifying the original in-place.
    for data_x, data_y in zip(X_retrain, Y_retrain):
         
        # Predict the current label to check if it matches the true label.
        pred, encoded_vector = collapse_probability(am_, data_x, position_table, feature_table, dim)
        # If prediction does not match the true label, adjust the associative memory.
        if pred != data_y:
            # Subtract the encoded vector from the incorrectly predicted class.
            am_[pred] = np.subtract(am_[pred], encoded_vector)
            # Add the encoded vector to the correct class.
            am_[data_y] = np.add(am_[data_y], encoded_vector)
    return am_


def save_model(associative_memory, position_table, feature_table, filepath):
    """
    Saves the trained associative memory model to disk.

    Args:
    - associative_memory (np.ndarray): The associative memory to be saved.
    - filepath (str): Path where the associative memory will be stored.

    Returns:
    - None
    """
    with open(filepath, 'wb') as f:
        pickle.dump([associative_memory, position_table, feature_table], f)
    f.close()
    print("Associative memory saved")


def load_model(filepath):
    """
    Loads a trained associative memory model from disk.

    Args:
    - filepath (str): Path to the file from which the associative memory should be loaded.

    Returns:
    - np.ndarray: The associative memory loaded from the file.
    """
    with open(filepath, 'rb') as f:
        associative_memory, position_table, feature_table = pickle.load(f)
    print("Associative memory loaded")
    f.close()
    return associative_memory, position_table, feature_table

def quanitize():
    pass

def main(mode, epochs=3):
    """
    Main function to orchestrate the loading of the MNIST dataset, training, testing,
    retraining (if necessary), and saving/loading the model.
    """

    model_filepath = "mnist_associative_memory.pkl"
    print("Starting main function...")
    # Load MNIST dataset.
    mnist_path = "MNIST-csv"
    X_train, Y_train, X_test, Y_test = load_dataset(mnist_path)

    # Assume position_table and feature_table are generated here.
    dim = 10000  # Example dimensionality.

    if mode == 'train':
        position_table = create_data_representation(dim, 'bipolar', 28*28)
        feature_table = create_data_representation(dim, 'bipolar', 256)
        # Train the model.
        associative_memory = np.zeros((10, dim))  # Example initialization for 10 classes.
        associative_memory = train_model(associative_memory, X_train, Y_train, position_table, feature_table, dim)
        print("Training model done")
        save_model(associative_memory, position_table, feature_table, model_filepath)
        # Test the model.

    if mode == 'retrain':
        associative_memory, position_table, feature_table = load_model(model_filepath)
        for epoch in range(epochs):
            print("retraining. Currect epoch: ", epoch)
            associative_memory = retrain_model(associative_memory, X_train, Y_train, position_table, feature_table, dim)
        save_model(associative_memory, position_table, feature_table, model_filepath)
    
    if mode == 'test':
        associative_memory, position_table, feature_table = load_model(model_filepath)
        accuracy = test_data(associative_memory, X_test, Y_test, position_table, feature_table, dim)
        print(f"Test accuracy: {accuracy}")
    # Retrain the model if needed.
    # Example retraining data: X_retrain, Y_retrain (to be defined based on your needs).
    # associative_memory = retrain_model(associative_memory, X_retrain, Y_retrain, position_table, feature_table, dim)

    # Load the model.
    #loaded_associative_memory = load_model(model_filepath)

    # Optionally, quantize the loaded model for optimization.
    # Example target bit width for quantization (to be defined based on your needs).
    # quantized_associative_memory = quantize(loaded_associative_memory, target_bit_width=8)

def setup_cli_args():
    parser = argparse.ArgumentParser(
        description='This script allows training, retraining, and testing an associative memory model on the MNIST dataset. Use "train" mode to initially train the model, "retrain" mode to further train the model with additional epochs, and "test" mode to evaluate the model accuracy on the test dataset.',
        epilog='Example usage: python script_name.py train --epochs 5 (to train or retrain the model) or python script_name.py test (to test the model)'
    )
    parser.add_argument('mode', choices=['train', 'retrain', 'test'], help='Mode to run the script in. Options are "train", "retrain", or "test".')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for retraining. Only applicable in retrain mode.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = setup_cli_args()
    main(args.mode, args.epochs)
