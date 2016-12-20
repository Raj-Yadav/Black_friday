import numpy as np
from scipy.stats import mstats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("train.csv")

train.head(2)
train.isnull().sum()  # Product_category 2 and 3 have lots of NAN value.


train['Product_Category_2'].fillna(mstats.mode(train["Product_Category_2"]).mode[0], inplace = True)

train['Product_Category_3'].fillna(mstats.mode(train["Product_Category_3"]).mode[0], inplace = True)

train.isnull().sum()  ## No NAN value present in train data
#train = train[train.Purchase <= 21000]
train.drop("Product_ID", axis = 1, inplace = True)
train.drop("User_ID", axis = 1, inplace = True)

# # Data Exploration #############
train.columns
sns.boxplot(train["Purchase"])
sns.boxplot(x = "Age", y = "Purchase", data = train) # distribution is simple
sns.boxplot(x = "City_Category", y = "Purchase", data = train)
sns.boxplot(x = "Stay_In_Current_City_Years", y = "Purchase", data = train)

#train = pd.get_dummies(train)
#train.columns

y = np.array(train['Purchase'])
X = train.drop("Purchase", axis = 1)
X = pd.get_dummies(X)
y.shape
### Applying Machine Learning Technique
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.feature_selection import SelectKBest , f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
reg = LinearRegression()
scale =  StandardScaler()
pca = PCA(n_components = 2)
#X = scale.fit_transform(X)

#X_new = SelectKBest(f_regression, k=1).fit_transform(X,y)
X_pca = pca.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.30, random_state = 4)

reg.fit(X_train, y_train)
pred = reg.predict(X_test)

score = np.sqrt(mean_squared_error(y_test, pred))
scorer = r2_score(y_test, pred)
print("Scores for linear regression is {}".format(score))

### SGDRegressor  #####

from sklearn.linear_model import SGDRegressor
clf = SGDRegressor()
#clf.fit(X_train,y_train)
score = cross_val_score(clf,X_pca, y, scoring = 'neg_mean_squared_error', cv = 5)
#pred = clf.predict(X_test)
scores = np.sqrt(-np.mean(score))

print("Scores for linear regression is {}".format(scores))

clf.fit(X_pca, y)
##########

data = pd.read_csv("test.csv")
data.head(2)
data.isnull().sum()

data['Product_Category_2'].fillna(mstats.mode(data["Product_Category_2"]).mode[0], inplace = True)
data['Product_Category_3'].fillna(mstats.mode(data["Product_Category_3"]).mode[0], inplace = True)


sub = data.loc[:,["User_ID", "Product_ID"]]
data.drop("Product_ID", axis = 1, inplace = True)
data.drop("User_ID", axis = 1, inplace = True)
data.dtypes
data = pd.get_dummies(data)
data.head(1)
data = pca.fit_transform(data)
pred = clf.predict(data)
pred = pd.DataFrame(pred)
pred.shape
frame = [sub , pred]

result = pd.concat(frame, axis = 1)
result.shape
result.columns = ["User_ID", "Product_ID", "Purchase"]
result.head(1)

###### getting output file
result.to_csv("output1.csv",sep = ",")



    