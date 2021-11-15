import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
path1 = 'data/hepatitis.csv'
path2 = 'data/bankrupcy.csv'

# Load data from data file.
hepatitis_data = pd.read_csv(path1)
bankrupcy_data = pd.read_csv(path2)

# Print the distribution of the class of the data
plt.figure(figsize=(16,6))
plt.subplot(131), sns.countplot(x='ClassLabel', data=hepatitis_data)
plt.title('Distribution of the two classes (Hepatitis)')
plt.subplot(133), sns.countplot(x='ClassLabel', data=bankrupcy_data)
plt.title('Distribution of the two classes (Bankruptcy)')
plt.show()

def shuffle_data(path):
    # Load data from data file.
    shuffled_data = pd.read_csv(path)
    shuffled_data.insert(0, column='Bias', value=1)
    return shuffled_data.sample(frac=1)


class LogisticRegression:

    def __init__(self, data, folds, max_iter=10000, beta=0.99, reg_term=0):
        self.data = data
        self.folds = folds
        self.max_iter = max_iter
        self.beta = beta
        self.reg_term = reg_term

        self.cv_acc, self.select_line, self.cv_acc_mean = 0, 0, 0
        self.gradient_values, self.train_acc_values, self.cv_acc_values = [], [], []

    def set_learning_rate(self, lr):
        self.lr = lr

    def partition(self, fold):
        data = self.data
        # to exclude last term in previous partition for training data
        train_add = 1 if fold < self.folds else 0
        # to exclude last term in previous partition for testing data
        test_add = 1 if fold > 0 else 0

        # number of data sets
        n = len(self.data)
        test_set = data.iloc[int((fold) / self.folds * n + test_add):int((fold + 1) / self.folds * n), :]

        train_set_1 = data.iloc[int((fold + 1) / self.folds * n) + train_add:n, :]
        train_set_2 = data.iloc[0:int((fold) / self.folds * n), :]
        train_set = pd.concat([train_set_1, train_set_2])

        train_X = train_set.iloc[:, :-1].values
        train_y = train_set.iloc[:, -1].values
        train_y = np.reshape(train_y, (-1, 1))

        test_X = test_set.iloc[:, :-1].values
        test_y = test_set.iloc[:, -1].values
        test_y = np.reshape(test_y, (-1, 1))

        return train_X, train_y, test_X, test_y

    # XwT2 calculates the dot product of X and theta raised to power two. Then they are summed up and divided by 2*length of X and returned. we find the difference between predicted values
    # and the original y values and sum them up and find the average and return the cost
    '''
    def cost_cal(self, X, y, theta):
        XwT2 = np.power(((X @ theta.T) - y), 2)
        cost = np.sum(XwT2) / (2 * len(X))
        return cost
    '''

    def fit(self, X, y, v_X, v_y, reg_term=0.5,
            epsilon=5e-3):  # attempted termination condition - lack of improvement in cross validation set
        # Retrive the learning rate
        lr = self.lr
        # Retrive the maximum iteration
        max_iter = self.max_iter
        # Retrive beta
        beta = self.beta
        # Retrive reg_term
        reg_term = self.reg_term
        N, D = len(X[0]), len(X[0])
        w = np.zeros((N, 1))
        iterate, cv_acc, prev_cv_acc, d_theta = 0, 0, 0, 0
        max_cv_acc = 0  # maximum cross validation accuracy - records thetas at highest cv_acc
        best_theta = w
        g = np.inf

        while (np.linalg.norm(
                g) > epsilon):  # can add in 'or cv_acc>=prev_cv_acc-0.03' to stop when gradient becomes too small, 0.03 gives buffer
            g = self.gradient(X, y, w, reg_term)
            d_theta = (1 - beta) * g + beta * d_theta  # momentum
            w = w - lr * d_theta

            if iterate % 10 == 0:
                cv_pred = self.predict(v_X, w)
                prev_cv_acc = cv_acc
                cv_acc = self.accuracy(cv_pred, v_y)
                train_pred = self.predict(X, w)
                train_acc = self.accuracy(train_pred, y)
            if cv_acc >= max_cv_acc:  # checks if maximum accuracy thus far
                max_cv_acc = cv_acc
                best_theta = w
                self.select_line = iterate
            iterate += 1
            self.gradient_values.append(np.linalg.norm(g))
            self.train_acc_values.append(train_acc)
            self.cv_acc_values.append(cv_acc)
            #             if iterate % 100 == 0:
            #                 print(np.linalg.norm(g)/len(X))
            if iterate > max_iter:  # since it may not always converge, place a hard ceiling on number of iterations
                break

        # cost = self.cost_cal(X, y, best_theta)
        print(max_cv_acc)
        print(cv_acc)
        self.cv_acc = max_cv_acc
        self.cv_acc_mean = np.mean(self.cv_acc_values)
        return best_theta

    # def predict_proba(self, X, theta):
    #     return self.log_func(np.dot(X, theta))

    def predict(self, X, w):
        prediction = self.log_func(np.dot(X, w))
        predict_arr = []
        for i in prediction:
            if i >= 0.5:
                predict_arr.append(1)
            else:
                predict_arr.append(0)

        return predict_arr

    def accuracy(self, predict_arr, y):
        correct = 0
        for i, j in zip(predict_arr, y):
            if i == j[0]:
                correct += 1
        return correct / len(y)  # accuracy = # tp+tn / total

    def log_func(self, alpha):
        return 1 / (1 + np.exp(-alpha))

    def gradient(self, X, y, w, lambdaa):  # lambdaa is regularization term
        N = len(X[0])
        y_hat = self.log_func(np.dot(X, w))
        grad = np.dot(X.T, y_hat - y) / N
        grad[1:] += lambdaa * w[1:]
        return grad

    def get_test_acc(self, test_X, test_y, thetas):
        test_y = np.reshape(test_y, (-1, 1))

        return self.accuracy(self.predict(test_X, thetas), test_y)


folds = 10
accuracies = []
mean_acc = []
learning_rate = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

shuffled_data = shuffle_data(path2)
for lr in learning_rate:
    for fold in range(folds):
        log_reg = LogisticRegression(shuffled_data, folds)
        log_reg.set_learning_rate(lr)
        t_X, t_y, v_X, v_y = log_reg.partition(fold)
        # t_X --> test value X, v_X --> validation value X
        log_reg.fit(t_X, t_y, v_X, v_y)

        accuracies.append(log_reg.cv_acc_mean)
        plt.figure()
        plt.plot(log_reg.train_acc_values, label='Training accuracy')
        plt.plot(log_reg.cv_acc_values, label='CV accuracy')
        plt.axvline(log_reg.select_line, color='r', label='Weights Selected')
        plt.plot([], [], ' ', label="Learning Rate: " + str(log_reg.lr))
        plt.legend()
        plt.show()

        print("Learning Rate: " + str(lr))
        print("Fold: " + str(fold))
        print("Average Accuracy: " + str(np.mean(accuracies)))
    mean_acc.append(np.mean(accuracies))

plt.plot(learning_rate, mean_acc, 'o')
plt.xscale("log")
plt.xlabel("log(Learning Rate)")
plt.ylabel("Mean Accuracy")
plt.show()