import unittest
import pandas as pd
import numpy as np
from src.data_preprocessing import clean_data, normalize_data, transform_data, preprocess_protein_data


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 5],
            'feature2': [5, 4, 3, np.nan, 1, 1],
            'feature3': [9, 8, 7, 6, np.nan, 6]
        })


    def test_clean_data(self):
        cleaned_data = clean_data(self.sample_data)
        self.assertFalse(cleaned_data.isnull().values.any(), 
                         "Data contains null values after cleaning")
        self.assertEqual(len(cleaned_data), 4, 
                         "Data contains duplicates after cleaning")


    def test_normalize_data(self):
        cleaned_data = clean_data(self.sample_data)
        normalized_data = normalize_data(cleaned_data)
        self.assertTrue(np.allclose(normalized_data.mean(), 0, atol=1e-7), 
                        "Data is not normalized correctly")
        self.assertTrue(np.allclose(normalized_data.std(), 1, atol=1e-7), 
                        "Data is not normalized correctly")


    def test_transform_data(self):
        cleaned_data = clean_data(self.sample_data)
        transformed_data = transform_data(cleaned_data)
        self.assertTrue((transformed_data >= 0).all().all(), 
                        "Data contains negative values after transformation")


    def test_preprocess_protein_data(self):
        preprocessed_data = preprocess_protein_data(self.sample_data)
        self.assertFalse(preprocessed_data.isnull().values.any(), 
                         "Data contains null values after preprocessing")
        self.assertTrue(np.allclose(preprocessed_data.mean(), 0, atol=1e-7), 
                        "Data is not normalized correctly after preprocessing")
        self.assertTrue(np.allclose(preprocessed_data.std(), 1, atol=1e-7), 
                        "Data is not normalized correctly after preprocessing")
        self.assertTrue((preprocessed_data >= 0).all().all(), 
                        "Data contains negative values after preprocessing")


if __name__ == '__main__':
    unittest.main()
