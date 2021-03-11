#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello World!")


# In[124]:


import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"  #엠일이 아니고 엠엘이
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()


# In[125]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[126]:


housing = load_housing_data()
housing.head()


# In[127]:


housing.info()   #데이터에 대한 간략한 설명과 전체 행 수, 타입, 값의 개수 확인 등


# In[128]:


housing["ocean_proximity"].value_counts()     #어떤 카테고리, 각자 얼마나 많은 구역


# In[129]:


housing.describe()     #숫자형 특성의 요약정보 보여줌


# In[130]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))        #히스토그램 그리기ㅎㅎ
plt.show()


# In[131]:


import numpy as np

def split_train_test(data, test_ratio):       #테스트 세트 만들기
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[132]:


train_set, test_set = split_train_test(housing, 0.2)  #20% 추출
len(train_set)


# In[133]:


len(test_set)


# In[134]:


from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[135]:


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[136]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[137]:


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])

housing["income_cat"].hist()


# In[138]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[139]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[140]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[141]:


#2.4 시작
housing = strat_test_set.copy()


# In[142]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[143]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[144]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s = housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()


# In[145]:


corr_matrix = housing.corr()


# In[146]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[147]:


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_bedrooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


# In[148]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# In[149]:


housing["room_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedroom_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]


# In[150]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[159]:


#2.5 시작
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[160]:


housing.dropna(subset=["total_bedrooms"]) # 옵션 1
housing.drop("total_bedrooms", axis=1) # 옵션 2
median = housing["total_bedrooms"].median() # 옵션 3
housing["total_bedrooms"].fillna(median, inplace=True)


# In[161]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")


# In[162]:


housing_num = housing.drop("ocean_proximity", axis=1)


# In[163]:


imputer.fit(housing_num)


# In[164]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[165]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[166]:


ordinal_encoder.categories_


# In[167]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[168]:


housing_cat_1hot.toarray()


# In[169]:


cat_encoder.categories_


# In[170]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[171]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[172]:


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)


# In[173]:


#2.6 시작
from sklearn.linear_model import LinearRegression
 
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[174]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측: ", lin_reg.predict(some_data_prepared))


# In[175]:


print("레이블: ", list(some_labels))


# In[188]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[189]:


from sklearn.tree import DecisionTreeRegressor
 
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[190]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[191]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[207]:


def display_scores(scores):
    print("점수:", scores)
    print("평균:", scores.mean())
    print("표준편차:", scores.std())
    
display_scores(tree_rmse_scores)


# In[208]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[204]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[209]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[ ]:


import joblib
 
joblib.dump(my_model, "my_model.pkl")
# 그리고 나중에...
my_model_loaded = joblib.load("my_model.pkl")

