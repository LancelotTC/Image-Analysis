from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

from sklearn.feature_extraction import DictVectorizer, FeatureHasher

X = [
    {"city": "NYC", "temp": 70, "is_raining": False},
    {"city": "LA", "temp": 85, "is_raining": False},
    {"city": "NYC", "temp": 60, "is_raining": True},
]

# DictVectorizer: learns a vocabulary and outputs dense/sparse matrix
vec = DictVectorizer(sparse=False)
X_vec = vec.fit_transform(X)
print("DictVectorizer feature names:", vec.get_feature_names_out())
print("DictVectorizer output:\n", X_vec)

# FeatureHasher: hashes features into a fixed-size space, no fit() needed
hasher = FeatureHasher(n_features=8, input_type="dict")
X_hash = hasher.transform(X)
print("FeatureHasher output (dense):\n", X_hash.toarray())
