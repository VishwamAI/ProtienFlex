import unittest
import numpy as np
import tensorflow as tf
from src.model_training import build_model, train_model, evaluate_model, train_and_evaluate_model


class TestModelTraining(unittest.TestCase):


    def setUp(self):
        # Sample data for testing
        self.sample_data = np.random.rand(100, 10, 10, 10, 1)
        self.sample_labels = np.random.randint(2, size=100)


    def test_build_model(self):
        input_shape = self.sample_data.shape[1:]
        model = build_model(input_shape)
        self.assertIsInstance(model, tf.keras.Model, 
                              "Model is not an instance of tf.keras.Model")
        self.assertEqual(model.input_shape[1:], input_shape, 
                         "Model input shape does not match expected shape")


    def test_train_model(self):
        input_shape = self.sample_data.shape[1:]
        model = build_model(input_shape)
        X_train, X_val = self.sample_data[:80], self.sample_data[80:]
        y_train, y_val = self.sample_labels[:80], self.sample_labels[80:]
        history = train_model(model, X_train, y_train, X_val, y_val, 
                              epochs=1, batch_size=10)
        self.assertIn('accuracy', history.history, 
                      "Training history does not contain 'accuracy'")
        self.assertIn('val_accuracy', history.history, 
                      "Training history does not contain 'val_accuracy'")


    def test_evaluate_model(self):
        input_shape = self.sample_data.shape[1:]
        model = build_model(input_shape)
        X_train, X_test = self.sample_data[:80], self.sample_data[80:]
        y_train, y_test = self.sample_labels[:80], self.sample_labels[80:]
        train_model(model, X_train, y_train, X_test, y_test, 
                    epochs=1, batch_size=10)
        evaluation_metrics = evaluate_model(model, X_test, y_test)
        self.assertIn('accuracy', evaluation_metrics, 
                      "Evaluation metrics do not contain 'accuracy'")
        self.assertIn('loss', evaluation_metrics, 
                      "Evaluation metrics do not contain 'loss'")


    def test_train_and_evaluate_model(self):
        evaluation_metrics = train_and_evaluate_model(self.sample_data, 
                                                      self.sample_labels, 
                                                      test_size=0.2, 
                                                      val_size=0.2, 
                                                      epochs=1, 
                                                      batch_size=10)
        self.assertIn('accuracy', evaluation_metrics, 
                      "Evaluation metrics do not contain 'accuracy'")
        self.assertIn('loss', evaluation_metrics, 
                      "Evaluation metrics do not contain 'loss'")


if __name__ == '__main__':
    unittest.main()
