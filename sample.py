import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from itertools import islice
import numpy
import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
numpy.random.seed(7)
SAMPLE_SIZE = 10
MODIFIER = 16

class EarlyStoppingByLoss(Callback):
    def __init__(self, monitor='loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current <= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def main():

	with open("sha1.txt", "r") as f:
		rawX = list(islice(f, SAMPLE_SIZE))

	rawX = [data.rstrip() for data in rawX]

	X = numpy.zeros((len(rawX), len(rawX[0])))

	for i in range(len(rawX)):
		for j in range(len(rawX[i])):
			X[i][j] = ord(rawX[i][j])

	norm =  numpy.linalg.norm(X)

	for i in range(len(X)):
		for j in range(len(X[i])):
			X[i][j] = X[i][j]/norm


	with open("md5.txt", "r") as f:
		rawY = list(islice(f, SAMPLE_SIZE))

	rawY = [data.rstrip() for data in rawY]

	Y = numpy.zeros((len(rawY), len(rawY[0])))

	for i in range(len(rawY)):
		for j in range(len(rawY[i])):
			Y[i][j] = ord(rawY[i][j])


	model = Sequential()
	model.add(Dense(len(X[0]), input_shape=(len(X[0]),), activation='relu'))
	model.add(Dense(len(X[0])*MODIFIER, activation='relu'))
	model.add(Dense(len(Y[0])*MODIFIER, activation='relu'))
	model.add(Dense(len(Y[0])))

	model.compile(loss='mean_squared_error', optimizer='adam')

	cb = EarlyStoppingByLoss()

	start_time = time.time()
	model.fit(X, Y, epochs=9999999999999999999, callbacks=[cb])
	stop_time = time.time()

	pred =  model.predict(X)

	print("Actual Values: ")
	print(Y)

	print("Predicted Values: ")
	print(numpy.round(pred))

	print("Time Training: " + str(stop_time-start_time) + " seconds")

	model.save("my_model.h5")

if __name__ == "__main__":
	main()


'''
model = load_model("my_model.h5")

pred =  model.predict(X)

print("Predicted Values: ")
print(pred)
'''
