from train_xgboost import *
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


if __name__ == '__main__':
    random.seed(1)

    model = xgboost.XGBRegressor()
    model.load_model('model.xgb')

    X_test = []
    Y_test = []
    Y_predict = []
    for _ in range(TEST_SIZE):
        x = get_x()
        y = get_y(x)
        X_test.append(x)
        Y_test.append(y)

        y_predict = model.predict([x])[0]
        Y_predict.append(y_predict)
        # print('y', y, y_predict)

    mae = mean_absolute_error(Y_test, Y_predict)
    print('mae', mae)

    f_importance = model.get_booster().get_score(importance_type='gain')
    print('f_importance', f_importance)

    plt.bar(*zip(*f_importance.items()))
    plt.show()

    X = []
    Y = []
    for i in range(1000):
        x = get_x()
        x[0] = i / 250 - 2.0
        x[1] = 0
        x[2] = 0
        x[3] = -1.0
        # x[4] = 0
        y = model.predict([x])[0]
        X.append(x[0])
        Y.append(y)

    plt.plot(X, Y)
    plt.grid()
    plt.show()