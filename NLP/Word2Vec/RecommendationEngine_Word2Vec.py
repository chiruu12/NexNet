import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
import heapq

# Ensure NLTK resources are downloaded
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
        self.model = self.train_model

    def preprocess_text(self, text):
        """
        Tokenize and remove stopwords from the text.

        Parameters:
        - text (str): The text to preprocess.
        """
        tokens = word_tokenize(re.sub(r'[^a-zA-Z\s]', '', text.lower()))
        return [token for token in tokens if token not in self.stop_words]

    def train_model(self, vector_size=100, window=5, min_count=1, workers=4):
        """
        Train the Word2Vec model using the specified text columns.

        Parameters:
        - vector_size (int): Dimensionality of the word vectors.
        - window (int): Maximum distance between the current and predicted word within a sentence.
        - min_count (int): Ignores all words with total frequency lower than this.
        - workers (int): Number of worker threads to train the model.
        """
        combined_text = self.df[self.text_columns].fillna('').agg(' '.join, axis=1)
        # Preprocess text and prepare corpus
        corpus = [self.preprocess_text(text) for text in tqdm(combined_text, desc="Preprocessing Text")]
        self.model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count,
                              workers=workers)
        print("Model training completed.")

    def get_item_vector(self, item_id):
        """
        Get the vector representation of an item by averaging the vectors of words in its text.

        Parameters:
        - item_id (str): The unique identifier of the item.
        """
        item_text = self.df.loc[self.df[self.item_id_col] == item_id, self.text_columns].fillna('').agg(' '.join,
                                                                                                        axis=1).values
        if len(item_text) == 0:
            return None
        tokens = self.preprocess_text(item_text[0])
        vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    def print_similar_items(self, similarities):
        """
        Prints the similar items along with their descriptions, information, and similarity scores.

        Parameters:
        - similarities (list): A list of tuples with item_id and similarity score.
        """
        for _, row in enumerate(similarities):
            item_id, similarity = row
            similar_item = self.df.loc[self.df[self.item_id_col] == item_id].squeeze()
            print(f"\nItem ID: {item_id}")
            print(f"Description: {similar_item['description']}")
            print(f"Information: {similar_item['text']}")
            print(f"Similarity Score: {similarity:.4f}")

    def recommend(self, target_item_id, top_n=5, print_similar: bool = False):
        """
        Recommend items similar to the target item.

        Parameters:
        - target_item_id (str): The unique identifier of the target item.
        - top_n (int): The number of similar items to recommend.
        - print_similar (bool): if True it will print the similar entities

        Returns:
        - DataFrame: A DataFrame containing recommended item IDs and their similarity scores.
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet. Please call train_model() first.")
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
                similarity = np.dot(target_vector, item_vector) / (
                            np.linalg.norm(target_vector) * np.linalg.norm(item_vector))

                if len(similarities) < top_n:
                    heapq.heappush(similarities, (similarity, item_id))
                else:
                    if similarity > similarities[0][0]:
                        heapq.heapreplace(similarities, (similarity, item_id))

        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
        if print_similar:
            print(f"Top {top_n} similar items to Item ID {target_item_id}:")
            self.print_similar_items(similarities)
        return pd.DataFrame(similarities, columns=[self.item_id_col, 'similarity'])
