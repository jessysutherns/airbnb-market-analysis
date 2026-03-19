import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

# =============================
# SETUP
# =============================
sns.set_style("whitegrid")
os.makedirs("results", exist_ok=True)

nltk.download('punkt')
nltk.download('stopwords')

# =============================
# LOAD DATA
# =============================
print("Loading data...")
properties = pd.read_csv("data/listings.csv")
reviews = pd.read_csv("data/reviews.csv")

print("Properties:", properties.shape)
print("Reviews:", reviews.shape)

# =============================
# CLEAN DATA
# =============================
print("Cleaning data...")

properties['price'] = properties['price'].str.replace(r'[^\d.]', '', regex=True).astype(float)
properties['price'] = properties['price'].fillna(properties['price'].median())

properties['last_review'] = pd.to_datetime(properties['last_review'], errors='coerce')
reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')

reviews['comments'] = reviews['comments'].astype(str).fillna('')
reviews['comments'] = reviews['comments'].str.lower().str.strip()

# =============================
# EDA - TOP NEIGHBOURHOODS
# =============================
print("Generating top neighbourhoods plot...")

top_neigh = properties['neighbourhood_cleansed'].value_counts().head(10)

plt.figure(figsize=(10,6))
top_neigh.plot(kind='bar')
plt.title("Top 10 Neighbourhoods")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/top_neighbourhoods.png")
plt.close()

# =============================
# ROOM TYPES
# =============================
print("Generating room types plot...")

plt.figure(figsize=(12,6))
sns.countplot(data=properties, x='neighbourhood_cleansed',
              hue='room_type',
              order=top_neigh.index)

plt.title("Room Types in Top Neighbourhoods")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/room_types.png")
plt.close()

# =============================
# RENTAL TYPES
# =============================
print("Generating rental types plot...")

properties['rental_type'] = properties['minimum_nights'].apply(
    lambda x: 'Short-Term' if x < 30 else 'Long-Term'
)

plt.figure(figsize=(12,6))
sns.countplot(data=properties, x='neighbourhood_cleansed',
              hue='rental_type',
              order=top_neigh.index)

plt.title("Rental Types in Top Neighbourhoods")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/rental_types.png")
plt.close()

# =============================
# TIME SERIES
# =============================
print("Generating time series plot...")

properties['year'] = properties['last_review'].dt.year

year_counts = properties.groupby('year').size()
year_price = properties.groupby('year')['price'].mean()

plt.figure(figsize=(10,6))
plt.plot(year_counts, label='Listings')
plt.plot(year_price, label='Average Price')
plt.legend()
plt.title("Listings & Price Trends")
plt.tight_layout()
plt.savefig("results/time_series.png")
plt.close()

# =============================
# NLP - LDA
# =============================
print("Running topic modelling...")

reviews_sample = reviews.sample(n=50000, random_state=42)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

reviews_sample['cleaned'] = reviews_sample['comments'].apply(clean_text)

vectorizer = CountVectorizer(max_df=0.95, min_df=2)
X = vectorizer.fit_transform(reviews_sample['cleaned'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\nTop 5 Topics:")
for i, topic in enumerate(lda.components_):
    topic_words = [words[i] for i in topic.argsort()[-10:]]
    print(f"Topic {i+1}: {', '.join(topic_words)}")

print("\nAnalysis complete. Results saved in 'results/' folder.")