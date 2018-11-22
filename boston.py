import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error

def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    print(train_data.info())
    print('-'*40)
    print(test_data.info())
    return train_data, test_data

def simple_train(train_data_X, train_data_y, test_data_X):
    X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_y, test_size=0.3, random_state=1)
    xg_reg = xgb.XGBRegressor(object='reg:linear', colsample_bylevel=0.3, learning_rate=0.1, max_depth=10, n_estimators=100,reg_lambda=2)
    xg_reg.fit(X_train,y_train)
    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_absolute_error(y_test,preds))
    print("Train RMSE: %f" % rmse)
    test_preds = xg_reg.predict(test_data_X)
    print(test_preds)



if __name__ == '__main__':
    train_data, test_data = load_data()
    train_data_X = train_data.drop(['ID','medv'], axis=1)
    train_data_Y = train_data['medv']
    test_data_X = test_data.drop(['ID'], axis=1)
    print(train_data_X.info())
    simple_train(train_data_X, train_data_Y,test_data_X)
