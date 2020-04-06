import numpy as np

class LogisticRegression :

    def __init__(self, learning_rate = None, loss=None) :

        # Global Initialization function

        if learning_rate == None :
            self.learning_rate = 0.005
        else :
            self.learning_rate = learning_rate
        self.costs = []

        if loss == None :
            self.loss_metric = 'binary_crossentropy'
        else :
            self.loss_metric = loss

        self.history = {}

    # Utility functions

    def __sigmoid(self, z):

        # This method computes the sigmoid of z, a scalar or numpy array of any size.

        s = 1 / (1 + np.exp(-z))

        return s

    def accuracy_score(self, y_pred, y_true):

        # This method computes the accuracy of predictions

        if self.loss_metric == 'binary_crossentropy' or self.loss_metric == 'categorical_crossentropy':
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
            return (100 - np.mean(np.abs(y_pred != y_true)) * 100)


    def loss(self, y_pred, y_true) :

        # This method computes the loss, given the predicted and true labels

        if self.loss_metric == 'binary_crossentropy' or self.loss_metric == 'categorical_crossentropy':
            cost = -((1/self.batch_size) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)))

        return cost

    # Main methods

    def __initialize_parameters(self, X, y, metrics) :

        # This method initializes both the weight and the bias matrices according
        # to the input's dimensions

        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.batch_size = X.shape[0]
        self.w = np.zeros((self.input_dim, self.output_dim))
        self.dw = np.zeros((self.input_dim, self.output_dim))
        self.b = np.zeros((1, self.output_dim))
        self.db = np.zeros((1, self.output_dim))
        for metric in metrics :
            self.history['train_' + metric] = []
            self.history['val_' + metric] = []

    def __propagate(self, X, y) :

        # This method takes care of the backward propagation phase

        y_pred = np.zeros((self.batch_size, self.output_dim))
        y_pred = self.__sigmoid(np.dot(X, self.w) + self.b)
        cost = self.loss(y_pred, y)
        self.dw = (1/self.batch_size) * np.dot(X.T, (y_pred - y))
        self.db = (1/self.batch_size) * np.sum(y_pred - y)
        cost = np.squeeze(cost)

        return cost

    def __print_metrics(self, X_train, y_train, X_val, y_val, i, cost, metrics) :

        # This method is called when the verbosity is an integer that denotes
        # after how many epochs we print the training and validation metrics


        to_print = "epoch " + str(i) + " ==============\n"
        to_print += "Cost : " + str(cost)

        #if X_val and y_val :

        y_pred_train = self.predict_proba(X_train)
        y_pred_val = self.predict_proba(X_val)

        if 'accuracy' in metrics :

            train_accuracy = self.accuracy_score(y_pred_train, y_train)
            self.history['train_accuracy'].append(train_accuracy)

            val_accuracy = self.accuracy_score(y_pred_val, y_val)
            self.history['val_accuracy'].append(val_accuracy)

        if 'loss' in metrics :

            train_loss = self.loss(y_pred_train, y_train)
            self.history['train_loss'].append(train_loss)

            val_loss = self.loss(y_pred_val, y_val)
            self.history['val_loss'].append(val_loss)

        to_print += " , "
        to_print += ''.join([metric + " : " + str(self.history[metric][-1]) + " , " for metric in self.history.keys()])

        print(to_print)


    def __optimize(self, X_train, y_train, X_val, y_val, epochs, metrics, verbose) :

        # The optimize method will take care of the iterations,

        for i in range(epochs) :
            cost = self.__propagate(X_train, y_train)
            self.w -= (self.learning_rate * self.dw)
            self.b -= (self.learning_rate * self.db)
            self.costs.append(cost)

            if verbose and i%verbose==0 :
                self.__print_metrics(X_train, y_train, X_val, y_val, i, cost, metrics)


    def fit(self, X_train, y_train, X_val, y_val, epochs=200, metrics=[], verbose=None) :

        # This is the method that we'll want to run to train the model
        # it will initialize the parameters then call the optimize method which will take
        # care of the forward and backward propagations

        #assert (not ((X_val and not y_val) or (not X_val and y_val)))
        assert (epochs > 0)

        if (len(y_train.shape) == 1) :
            y_placeholder = np.zeros((y_train.shape[0], 1))
            y_placeholder[:, 0] = y_train
            y_train = y_placeholder

        if (len(y_val.shape) == 1) :
            y_placeholder = np.zeros((y_val.shape[0], 1))
            y_placeholder[:, 0] = y_val
            y_val = y_placeholder

        self.__initialize_parameters(X_train, y_train, metrics)
        self.__optimize(X_train, y_train, X_val, y_val, epochs, metrics, verbose)

        return self.history


    def predict(self, X_test) :

        # This method will return 1-hot encoded predictions for the X_test labels

        y_pred = np.zeros((self.batch_size, self.output_dim))
        y_pred = self.__sigmoid(np.dot(X_test, self.w) + self.b)
        y_rounded = np.round(y_pred).astype(int)
        return y_rounded


    def predict_proba(self, X_test) :

        # This method will return the probabilities of the predictions of the labels
        # of X_test

        y_pred = np.zeros((self.batch_size, self.output_dim))
        y_pred = self.__sigmoid(np.dot(X_test, self.w) + self.b)
        return y_pred