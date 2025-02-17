from RecommendationEngine_Word2Vec import RecommendationEngine
import pandas as pd

# For dataset, you can download from here https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
# and use the Reviews.csv file
# although any dataset can be used for this, but I have used this one for demonstration
df = pd.read_csv('amazon-fine-food-reviews/Reviews.csv')
engine = RecommendationEngine(df, item_id_col='item_id', text_columns=['description', 'text'])
recommendations = engine.recommend(target_item_id='item1', top_n=3,print_similar=True)
