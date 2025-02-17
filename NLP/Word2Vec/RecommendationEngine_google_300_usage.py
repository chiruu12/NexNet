from RecommendationEngine_google_300 import RecommendationEngine
import pandas as pd
#Only use it if you have the GoogleNews-vectors-negative300.bin.gz file downloaded
# you can download it from here: https://www.kaggle.com/datasets/sugataghosh/google-word2vec


# for df you can download from here https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
# and use the Reviews.csv file
# although any dataset can be used for this, but I have used this one for demonstration
df = pd.read_csv('amazon-fine-food-reviews/Reviews.csv')
engine = RecommendationEngine(df, item_id_col='item_id', text_columns=['description', 'text'])
engine.load_pretrained_model("GoogleNews-vectors-negative300.bin.gz")
recommendations = engine.recommend(target_item_id='item1', top_n=3)
print(recommendations)
