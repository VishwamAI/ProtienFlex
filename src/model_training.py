import tensorflow as tf
from sklearn.model_selection import train_test_split


def build_model(input_shape):
    """
    Build a 3D model for protein folding using a neural network.
    
    Parameters:
    input_shape (tuple): The shape of the input data.
    
    Returns:
    tf.keras.Model: The constructed neural network model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train the 3D model using the training data.
    
    Parameters:
    model (tf.keras.Model): The neural network model to be trained.
    X_train (np.ndarray): The training data.
    y_train (np.ndarray): The training labels.
    X_val (np.ndarray): The validation data.
    y_val (np.ndarray): The validation labels.
    epochs (int): The number of epochs to train the model.
    batch_size (int): The batch size for training.
    
    Returns:
    tf.keras.callbacks.History: The training history of the model.
    """
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using the test data.
    
    Parameters:
    model (tf.keras.Model): The trained neural network model.
    X_test (np.ndarray): The test data.
    y_test (np.ndarray): The test labels.
    
    Returns:
    dict: The evaluation metrics of the model.
    """
    evaluation = model.evaluate(X_test, y_test)
    return dict(zip(model.metrics_names, evaluation))


def train_and_evaluate_model(data, labels, test_size=0.2, val_size=0.2, epochs=10, batch_size=32):
    """
    Train and evaluate the 3D model using the provided data and labels.
    
    Parameters:
    data (np.ndarray): The input data.
    labels (np.ndarray): The labels for the input data.
    test_size (float): The proportion of the data to be used as test data.
    val_size (float): The proportion of the training data to be used as validation data.
    epochs (int): The number of epochs to train the model.
    batch_size (int): The batch size for training.
    
    Returns:
    dict: The evaluation metrics of the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    
    train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    evaluation_metrics = evaluate_model(model, X_test, y_test)
    
    return evaluation_metrics
