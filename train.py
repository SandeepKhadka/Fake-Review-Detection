# import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import string
import warnings
import pickle
warnings.filterwarnings("ignore")

# import data
reviews_df = pd.read_csv("reviews_data.csv")
# Downloads the stopwords package
nltk.download('stopwords')
# turn CG/OG into numerical data, creating a new column called fake
reviews_df['fake'] = reviews_df['label'].apply(lambda x: 1 if x == 'OR' else 0)


def process_text(text):
    # 1 Remove punctuation
    # 2 Remove stopwords
    # 3 return list of clean text words

    # 1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # 2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    # 3
    return clean_words


# create train/test split
x_train, x_test, y_train, y_test = train_test_split(reviews_df.text_, reviews_df.fake, test_size = 0.20)

# creating pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer(analyzer=process_text)),
    ('nb', MultinomialNB())
])

# train model
clf.fit(x_train,y_train)

# store the trained model using pickle
pickle.dump(clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
