from keras import backend as K

def round_fullgrad(x):
	i = K.round(x)
	return x + K.stop_gradient(i-x)

def integerize(x, b):
	k = K.max(K.abs(x))
	return k/2**b, round_fullgrad(K.clip(x/k*2**b, -2**b, 2**b))
