import xgboost
import random, time

DIMENSION = 5
TRAIN_SIZE = 10000
TEST_SIZE = 100

def get_x():
    x = [random.uniform(-1, 1) for _ in range(DIMENSION)]
    x.append(x[0] * 3.0 + random.random() * 5.0)
    return x

def get_y(x):
    return x[0] ** 2 + x[1] * 0.5 + x[2] ** 3



if __name__ == '__main__':
    random.seed(2)

    model = xgboost.XGBRegressor(n_estimators=100, max_depth=6)

    X_train = []
    Y_train = []
    for _ in range(TRAIN_SIZE):
        x = get_x()
        y = get_y(x)
        # print('x', x, 'y', y)
        X_train.append(x)
        Y_train.append(y)

    model.fit(X_train, Y_train)
    model.save_model('model.xgb')

    # time.sleep(60)






