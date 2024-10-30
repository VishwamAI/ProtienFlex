import unittest
from src.nlp_processing import clean_text, tokenize_text, remove_stopwords, embed_text, process_text

class TestNLPProcessing(unittest.TestCase):

    def setUp(self):
        self.sample_text = "This is a sample text for testing NLP processing."

    def test_clean_text(self):
        cleaned_text = clean_text(self.sample_text)
        self.assertIsInstance(cleaned_text, str, "Cleaned text is not a string")
        self.assertNotIn('\W', cleaned_text, "Cleaned text contains special characters")
        self.assertNotIn('\s+', cleaned_text, "Cleaned text contains extra spaces")

    def test_tokenize_text(self):
        tokens = tokenize_text(self.sample_text)
        self.assertIsInstance(tokens, list, "Tokens are not in a list")
        self.assertTrue(all(isinstance(token, str) for token in tokens), "Not all tokens are strings")

    def test_remove_stopwords(self):
        tokens = tokenize_text(self.sample_text)
        filtered_tokens = remove_stopwords(tokens)
        stop_words = set(stopwords.words('english'))
        self.assertTrue(all(token.lower() not in stop_words for token in filtered_tokens), "Filtered tokens contain stopwords")

    def test_embed_text(self):
        tokens = tokenize_text(self.sample_text)
        filtered_tokens = remove_stopwords(tokens)
        embeddings = embed_text(filtered_tokens)
        self.assertIsInstance(embeddings, list, "Embeddings are not in a list")
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings), "Not all embeddings are numpy arrays")

    def test_process_text(self):
        embeddings = process_text(self.sample_text)
        self.assertIsInstance(embeddings, list, "Processed text embeddings are not in a list")
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings), "Not all processed text embeddings are numpy arrays")

if __name__ == '__main__':
    unittest.main()
