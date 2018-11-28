from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from itertools import islice
import matplotlib.pyplot as plt
import numpy
import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
numpy.random.seed(7)
SAMPLE_SIZE = 1000
MODIFIER = 16
HIDDEN = 1280
THRESHOLD = 0.01

class EarlyStoppingByLoss(Callback):
    def __init__(self, monitor='loss', value=THRESHOLD, verbose=0):
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

	'''norm =  numpy.linalg.norm(X)

	for i in range(len(X)):
		for j in range(len(X[i])):
			X[i][j] = X[i][j]/norm'''


	with open("md5.txt", "r") as f:
		rawY = list(islice(f, SAMPLE_SIZE))

	rawY = [data.rstrip() for data in rawY]

	Y = numpy.zeros((len(rawY), len(rawY[0])))

	for i in range(len(rawY)):
		for j in range(len(rawY[i])):
			Y[i][j] = ord(rawY[i][j])

	model = Sequential()
	model.add(Dense(len(X[0]), input_shape=(len(X[0]),), activation='elu'))
	#model.add(Dense(len(X[0])*MODIFIER, activation='elu'))
	#model.add(Dense(len(X[0])*len(Y[0]), activation='elu'))
	#model.add(Dense(len(Y[0])*MODIFIER, activation='elu'))
	model.add(Dense(HIDDEN, activation='elu'))
	model.add(Dense(len(Y[0]), activation='linear'))

	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
	model.compile(loss='mean_squared_error', optimizer=adam)

	cb = EarlyStoppingByLoss()

	start_time = time.time()
	history = model.fit(X, Y, epochs=9999999999999999999, callbacks=[cb])
	stop_time = time.time()

	pred =  model.predict(X)

	total = 0
	correct = 0

	for i in range(len(Y)):
		wrong = False
		for j in range(len(Y[0])):
			if Y[i][j] != numpy.round(pred[i][j]):
				wrong = True
				break
		if not wrong:
			correct += 1
		total += 1

	print("Precision: " + str(correct) + "/" +str(total))

	print("Time Training: " + str(stop_time-start_time) + " seconds")

	model.save("my_model.h5")

	plt.plot(history.history['loss'])
	plt.title('Loss Decay')
	plt.ylabel('Loss')
	plt.ylim(0,500)
	plt.xlabel('Epoch')
	plt.show()

if __name__ == "__main__":
	main()
''

'''
model = load_model("my_model.h5")

pred =  model.predict(X)

print("Predicted Values: ")
print(pred)
'''
