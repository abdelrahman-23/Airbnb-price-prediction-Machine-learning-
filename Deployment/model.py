# import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeRegressor

# Load the csv file after preprocessing
df = pd.read_csv("airbnb_preprocesses.csv")
print(df.head())

# choose columns
split_data = df[
    ['host_identity_verified', 'neighbourhood_group', 'instant_bookable', 'cancellation_policy', 'room type',
     'Construction year', 'minimum nights', 'number of reviews', 'reviews per month',
     'review rate number', 'calculated host listings count',
     'availability 365', 'price', 'service fee']]
# Select independent and dependent variable
x = split_data.drop(["price"], axis=1).values
y = split_data['price'].values
# Split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# 25 % --> test
# 75 % --> train

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Decision Tree Model
tdt = DecisionTreeRegressor().fit(x_train, y_train)
tdtv = tdt.score(x_train,y_train)
print("The Accuracy of the model is : " ,tdtv)

# Make pickle file of our model
pickle.dump(tdt, open("model.pkl", "wb"))
