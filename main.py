import math
import sys


class Perceptron:
    def __init__(self, columns, classes, rows):
        self.columns = columns
        self.classes: list = classes
        self.rows = rows
        self.weights = [0] * len(self.columns)
        self.outputs = [0] * len(self.rows)


def print_iteration(columns, count, weights, output):
    print_str = "After iteration " + str(count + 1) + ": "
    f1 = "w({}) = {:.4f}, "
    f2 = "output = {:.4f}"
    for i in range(len(weights)):
        print_str += f1.format(columns[i], weights[i])
    print_str += f2.format(output)
    print(print_str)


def train_perceptron(perceptron, iterations, alpha):
    idx = 0
    for i in range(iterations):
        y = perceptron.classes[idx]
        x = perceptron.rows[idx]
        update_weights(x, perceptron.weights, alpha, y)
        perceptron.outputs[idx] = sigmoid(dot_product(perceptron.weights, x))
        print_iteration(perceptron.columns, i, perceptron.weights, perceptron.outputs[idx])
        idx += 1
        if idx % len(perceptron.rows) == 0 and idx != 0:
            idx = 0


def run_test(test_table, perceptron):
    test_classification = []
    for row in test_table.rows:
        if sigmoid(dot_product(perceptron, row)) < 0.5:
            test_classification.append(0)
        else:
            test_classification.append(1)
    return test_classification


def calculate_accuracy(actual, predicted):
    count = 0
    for i in range(len(actual)):
        if (predicted[i] < 0.5 and actual[i] == 0) or (predicted[i] >= 0.5 and actual[i] == 1):
            count += 1
    return count / len(actual) * 100


def sigmoid(t):
    return 1 / (1 + math.exp(-t))


def dot_product(w, x):
    prod = 0
    for i in range(len(w)):
        prod += w[i] * x[i]
    return prod


def update_weights(x, weights, alpha, y):
    wx = dot_product(weights, x)
    sigmoid_wx = sigmoid(wx)
    delta = y - sigmoid_wx
    d_sigmoid = sigmoid_wx * (1 - sigmoid_wx)
    const_scale = alpha * delta * d_sigmoid
    for i in range(len(weights)):
        weights[i] = weights[i] + const_scale * x[i]


def read_file(filename):
    file = open(filename, "r")
    columns: str = file.readline()
    file_lines = file.readlines()
    columns: list = columns.split("\t")
    columns.pop()
    rows = []
    classes = []
    for line in file_lines:
        line = line.split("\t")
        classification = int(line.pop())
        int_line = []
        for value in line:
            int_line.append(int(value))
        rows.append(int_line)
        classes.append(classification)
    return Perceptron(columns, classes, rows)


if __name__ == '__main__':
    n = len(sys.argv)
    if n < 5:
        print("python3 main.py <training file> <test file> <learning rate> <iterations>")
        quit()
    elif n == 5:
        training_file = sys.argv[1]
        test_file = sys.argv[2]
        learning_rate = sys.argv[3]
        no_of_iterations = sys.argv[4]
        str_f = "\nAccuracy on {} set ({} instances): {:.1f}%"
        training_data = read_file(training_file)
        train_perceptron(training_data, int(no_of_iterations), float(learning_rate))
        test_results = run_test(training_data, training_data.weights)
        accuracy = calculate_accuracy(training_data.classes, test_results)
        print(str_f.format("training", len(training_data.rows), accuracy))
        test_data = read_file(test_file)
        test_results = run_test(test_data, training_data.weights)
        accuracy = calculate_accuracy(test_data.classes, test_results)
        print(str_f.format("test", len(test_data.rows), accuracy))
