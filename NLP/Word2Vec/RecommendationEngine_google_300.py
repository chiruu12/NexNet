import pandas as pd
import numpy as np
import re
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
import heapq

nltk.download('stopwords')
nltk.download('punkt')

class RecommendationEngine:
    def __init__(self, df, item_id_col, text_columns):
        """
        Initialize the recommendation engine with the provided DataFrame.

        Parameters:
        - df (DataFrame): DataFrame containing item data.
        - item_id_col (str): Column name for unique item identifiers.
        - text_columns (list): List of column names to be used for training the Word2Vec model.
        """
        self.df = df
        self.item_id_col = item_id_col
        self.text_columns = text_columns
        self.stop_words = set(stopwords.words('english'))
        self.model = None

    def preprocess_text(self, text):
        """
        Tokenize and remove stopwords from the text.

        Parameters:
        - text (str): The text to preprocess.

        Returns:
        - list: A list of processed tokens.
        """
        if not isinstance(text, str) or text.strip() == '':
            return []
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        return [token for token in tokens if token not in self.stop_words]

    def load_pretrained_model(self, model_path="GoogleNews-vectors-negative300.bin.gz"):
        """
        Load Google's pre-trained Word2Vec model.

        Parameters:
        - model_path (str): Path to the pre-trained model.
        """
        print("Loading pre-trained Word2Vec model...")
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print("Model loaded successfully.")

    def get_item_vector(self, item_id):
        """
        Get the vector representation of an item by averaging the vectors of words in its text.

        Parameters:
        - item_id (str): The unique identifier of the item.

        Returns:
        - np.array: The vector representation of the item.
        """
        item_text = self.df.loc[self.df[self.item_id_col] == item_id, self.text_columns].fillna('').agg(' '.join, axis=1).values
        if len(item_text) == 0:
            return None
        tokens = self.preprocess_text(item_text[0])

        vectors = []
        for token in tokens:
            try:
                vectors.append(self.model.get_vector(token))
            except KeyError:
                continue

        return np.mean(vectors, axis=0) if vectors else None

    def recommend(self, target_item_id, top_n=5):
        """
        Recommend items similar to the target item.

        Parameters:
        - target_item_id (str): The unique identifier of the target item.
        - top_n (int): The number of similar items to recommend.

        Returns:
        - DataFrame: A DataFrame containing recommended item IDs and their similarity scores.
        """
        if self.model is None:
            raise ValueError("The model has not been loaded yet. Please call load_pretrained_model() first.")

        target_vector = self.get_item_vector(target_item_id)
        if target_vector is None:
            print(f"Item ID {target_item_id} not found or has insufficient text data.")
            return pd.DataFrame(columns=[self.item_id_col, 'similarity'])

        similarities = []
        for item_id in tqdm(self.df[self.item_id_col].unique(), desc="Calculating Similarities"):
            if item_id == target_item_id:
                continue
            item_vector = self.get_item_vector(item_id)
            if item_vector is not None:
                similarity = np.dot(target_vector, item_vector) / (np.linalg.norm(target_vector) * np.linalg.norm(item_vector))

                if len(similarities) < top_n:
                    heapq.heappush(similarities, (similarity, item_id))
                else:
                    if similarity > similarities[0][0]:
                        heapq.heapreplace(similarities, (similarity, item_id))

        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
        return pd.DataFrame(similarities, columns=[self.item_id_col, 'similarity'])