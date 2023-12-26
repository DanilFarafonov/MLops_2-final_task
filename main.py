import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from clearml import Task, Logger

task = Task.init(project_name='MLOps-2.4', task_name='SVC hyperparameters')
logger = task.get_logger()

parameters = {
        'gamma': 0.5,
        'C': 2.0
    }

parameters = task.connect(parameters)

train = pd.read_csv('data/train.csv')
train.drop(['keyword', 'location'], axis=1, inplace=True)

nltk.download('stopwords')
nltk.download('wordnet')

stopwords_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()
ps = PorterStemmer()

def preprocess_data(data, stopwords, ps, wnl):

  # Приведение символов к нижнему регистру
  data = str(data).lower()
  # Удаление пробелов в начале и в конце
  data = data.strip()
  # Удаление знаков пунктуации
  data = re.sub(r'[^\w\d\s]', ' ', data)
  # Удаление веб-адресов
  data = data.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', ' ')
  # Удаление чисел
  data = re.sub(r'\d+', ' ', data)
  # Удаление лишних пробелов (замена нескольких идущих подряд пробелов на один)
  data = data.replace(r'\s+', ' ')
  # Токенизация
  data_token = data.split()
  # Удаление стоп-слов из списка
  data_token = [word for word in data_token if word not in stopwords]
  # Стемминг
  data_token = [ps.stem(word) for word in data_token]
  # Лемматизация
  data_token = [wnl.lemmatize(word) for word in data_token]
  # Перевод данных обратно из списка в строку
  data = ' '.join(data_token)
  return data

train['clean'] = train['text'].apply(lambda x: preprocess_data(x, stopwords_list, ps, wnl))
X_train, X_val, y_train, y_val = train_test_split(train['clean'], train['target'], test_size=0.20, random_state=42)

X_train_tfidf = train['clean']
tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
tfidf_vectorizer.fit(X_train_tfidf)

X_train = tfidf_vectorizer.transform(X_train)
X_val = tfidf_vectorizer.transform(X_val)

svc = SVC(
    C=parameters["C"],
    gamma=parameters["gamma"],
    kernel="rbf"
)

svc.fit(X_train, y_train)
y_predict = svc.predict(X_val)
f1 = f1_score(y_val, y_predict)

Logger.current_logger().report_scalar(title='F1_score', series='F1', value=f1, iteration=1)
task.upload_artifact(name='F1', artifact_object={'F1': f1})
