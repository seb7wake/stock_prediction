import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

TRAIN_DATA = '../CSV_Files/Bitcoin Year Data.csv'
TEST_DATA = '../CSV_Files/Bitcoin Month Data.csv'

current_train_data = TRAIN_DATA
current_test_data = TEST_DATA

NUM_TRAIN_DATA_POINTS = len(pd.read_csv(TRAIN_DATA))-3
NUM_TEST_DATA_POINTS = len(pd.read_csv(TEST_DATA))-3

def load_stock_data(stock_name, num_data_points):
    data = pd.read_csv(stock_name, 
                       skiprows=0, 
                       nrows=num_data_points, 
                       usecols=['Price', 'Open', 'Vol.'])
    final_prices = data['Price'].astype(str).str.replace(',', '').astype(np.float)
    opening_prices = data['Open'].astype(str).str.replace(',', '').astype(np.float)
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_prices, opening_prices, volumes

def calculate_price_differences(final_prices, opening_prices):
    changes = []
    for i in range(len(opening_prices)-1):
        changes.append(opening_prices[i+1]-final_prices[i])
    return changes

def calculate_accuracy(expected, actual):
    correct = 0
    for i in range(len(actual)):
        if (actual[i] < 0 and expected[i] < 0) or (actual[i] < 0 and expected[i] < 0):
            correct += 1
    return (correct/len(actual))*100

# TRAINING DATASETS
train_finals, train_openings, train_volumes = load_stock_data(TRAIN_DATA, NUM_TRAIN_DATA_POINTS)
train_changes = calculate_price_differences(train_finals, train_openings)
train_volumes = train_volumes[:-1]

# TESTING DATASETS
test_finals, test_openings, test_volumes = load_stock_data(TEST_DATA, NUM_TEST_DATA_POINTS)
test_changes = calculate_price_differences(test_finals, test_openings)
test_volumes = test_volumes[:-1]

m = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)
b = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)

# used to input volumes
x = train_volumes

# each price change corresponds to a particular volume
# expected values -> used for training
y_predicted = train_changes

def loss():
    y = m*x + b
    # Outputs the cost (how inaccurate) the models prediction was
    # compared to what it should actually be
    cost = tf.reduce_sum(tf.square(y - y_predicted))
    return cost

# optimizer aimed at minimizing loss by changing W an b
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

# number of epochs is 100
for i in range(10000):
    train = optimizer.minimize(loss=loss, var_list=[m, b])

tf.print(m, b)
res = m*test_volumes + b
accuracy = calculate_accuracy(test_changes, res)
print('Accuracy of model:', accuracy, '%')

# plt.figure(1)
# plt.plot(train_volumes, train_changes, 'bo')
# plt.title('Price Differences for Given Volumes for the past Year')
# plt.xlabel('Volumes')
# plt.ylabel('Price Differences')
# plt.show()