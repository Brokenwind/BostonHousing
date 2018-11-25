import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
from sklearn import preprocessing

def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    print(train_data.info())
    print('-'*40)
    print(test_data.info())
    return train_data, test_data

def xgboost_train(train_data_X, train_data_y, test_data_X):
    X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_y, test_size=0.3, random_state=1)
    xg_reg = xgb.XGBRegressor(object='reg:linear', colsample_bylevel=0.3, learning_rate=0.1, max_depth=10, n_estimators=100,reg_lambda=2)
    xg_reg.fit(X_train,y_train)
    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_absolute_error(y_test,preds))
    print("Train RMSE: %f" % rmse)
    test_preds = xg_reg.predict(test_data_X)
    res_df = pd.DataFrame({'ID':test_data['ID'], 'medv':test_preds})
    res_df.to_csv('xgb_submission.csv',index=False)


def show1(train_data):
    sns.distplot(train_data['medv'])
    plt.show()

def show2(train_data):
    sns.lmplot(data=train_data,x='black',y='medv',hue='chas')
    plt.show()

def show3(train_data):
    sns.scatterplot(data=train_data,x='tax',y='medv')
    plt.show()

def regularization(combined_train_test):
    cols = ['crim','zn','indus','nox','nox','rm','age','dis','rad','tax','ptratio','black','lstat']
    scaler = preprocessing.StandardScaler().fit(combined_train_test[cols])
    combined_train_test[cols] = scaler.transform(combined_train_test[cols])

    return combined_train_test


if __name__ == '__main__':
    train_data, test_data = load_data()
    train_data_X = train_data.drop(['ID','medv'], axis=1)
    train_data_Y = train_data['medv']
    test_data_X = test_data.drop(['ID'], axis=1)
    combined_train_test = pd.concat([train_data_X,test_data_X],axis=0)
    print(train_data_X.describe())
    combined_train_test = regularization(combined_train_test)
    xgboost_train(train_data_X, train_data_Y, test_data_X)
    print(combined_train_test.describe())
    reg_train_data_X = combined_train_test[0:333]
    reg_test_data_X = combined_train_test[333:]
    xgboost_train(reg_train_data_X, train_data_Y, reg_test_data_X)
    #show2(train_data)
    #show3(train_data)