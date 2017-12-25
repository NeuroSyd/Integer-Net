import os
from keras.models import Sequential, Model
from keras.layers import Merge, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,Adagrad,Adadelta,RMSprop
from keras.constraints import max_norm
from keras import backend as K

from models.customCallbacks import MyEarlyStopping, MyModelCheckpoint
from models.xnor_layers import XnorConv2D, XnorDense
from models.binary_layers import BinaryConv2D, BinaryDense
from models.int_layers import IntConv2D, IntDense

class ConvNN(object):
	def __init__(self,target,batch_size=16,nb_classes=2,epochs=2,mode='cv',model='full'):
		self.target = target
		self.batch_size = batch_size
		self.nb_classes = nb_classes
		self.epochs = epochs
		self.mode = mode
		self.modelname = model
		assert mode in ['cv','test']

	def setup(self,X_train_shape):
		print ('X_train shape', X_train_shape)

		# Input shape = (None,1,16,200)
		inputs = Input(shape=X_train_shape[1:])

		normal1 = BatchNormalization(axis=1, name='normal1')(inputs)
		conv1 = Convolution2D(
			16,(X_train_shape[2],5),
			padding='valid', strides=(1,2),
			name='conv1')(normal1)
		relu1 = Activation('relu')(conv1)
		pool1 = MaxPooling2D(pool_size=(1,2))(relu1)

		normal2 = BatchNormalization(axis=1, name='normal2')(pool1)

		conv2 = Convolution2D(
			32, (1, 3),
			padding='valid', strides=(1,2),
			name='conv2')(normal2)
		relu2 = Activation('relu')(conv2)
		pool2 = MaxPooling2D(pool_size=(1,2))(relu2)

		normal3 = BatchNormalization(axis=1, name='normal3')(pool2)

		conv3 = Convolution2D(
			64, (1, 3),
			padding='valid', strides=(1,1),
			name='conv3')(normal3)
		relu3 = Activation('relu')(conv3)
		pool3 = MaxPooling2D(pool_size=(1,2))(relu3)

		flat = Flatten()(pool3)

		drop1 = Dropout(0.5)(flat)

		dens1 = Dense(128, activation='sigmoid', name='dens1')(drop1)
		drop2 = Dropout(0.5)(dens1)
		dens2 = Dense(self.nb_classes, name='dens2')(drop2)
		# option to include temperature in softmax
		temp = 1.0
		temperature = Lambda(lambda x: x / temp)(dens2)
		last = Activation('softmax')(temperature)

		self.model = Model(input=inputs, output=last)

		adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		self.model.compile(
			loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])

		return self

	def fit(self,X_train,Y_train,X_val=None, y_val=None):
		Y_train = Y_train.astype('uint8')
		Y_train = np_utils.to_categorical(Y_train, self.nb_classes)
		y_val = np_utils.to_categorical(y_val, self.nb_classes)

		early_stop = MyEarlyStopping(patience=10, verbose=0)

		filename = "weights_%s_%s_%s.h5" %(self.target, self.mode, self.modelname)
		checkpointer = MyModelCheckpoint(filename,
            verbose=0, save_best_only=True)

		if (y_val is None):
			self.model.fit(X_train, Y_train, batch_size=self.batch_size,
                           epochs=self.epochs,validation_split=0.2,
                           callbacks=[early_stop,checkpointer], verbose=2
                           )
		else:
			self.model.fit(X_train, Y_train, batch_size=self.batch_size,
						   epochs=self.epochs,validation_data=(X_val,y_val),
						   callbacks=[early_stop,checkpointer], verbose=2
						   )
		self.model.load_weights(filename)
		if self.mode == 'cv':
			os.remove(filename)
		return self

	def load_trained_weights(self, filename):
		self.model.load_weights(filename)
		print ('Loading pre-trained weights from %s.' %filename)
		return self

	def predict_proba(self,X):
		return self.model.predict([X])

	def evaluate(self, X, y):
		predictions = self.model.predict(X, verbose=0)[:,1]
		from sklearn.metrics import roc_auc_score
		auc_test = roc_auc_score(y, predictions)
		print('Test AUC is:', auc_test)
		return auc_test


class ConvNNXNOR(ConvNN):
	def setup(self,X_train_shape):
		print ('X_train shape', X_train_shape)

		# Input shape = (None,1,16,200)
		inputs = Input(shape=X_train_shape[1:])

		normal1 = BatchNormalization(axis=2, name='normal1')(inputs)
		conv1 = XnorConv2D(
			16, kernel_size=(X_train_shape[2],5), H=1.0,
			padding='valid', strides=(1,2), use_bias=False,
			name='conv1')(normal1)
		relu1 = Activation('relu')(conv1)
		pool1 = MaxPooling2D(pool_size=(1,2))(relu1)
        
		normal2 = BatchNormalization(axis=1, name='normal2')(pool1)
        
		conv2 = XnorConv2D(
			32, kernel_size=(1, 3), H=1.0,
			padding='valid', strides=(1,2), use_bias=False,
			name='conv2')(normal2)
		relu2 = Activation('relu')(conv2)
		pool2 = MaxPooling2D(pool_size=(1,2))(relu2)
        
		normal3 = BatchNormalization(axis=1, name='normal3')(pool2)
        
		conv3 = XnorConv2D(
			64, kernel_size=(1, 3), H=1.0,
			padding='valid', strides=(1,1), use_bias=False,
			name='conv3')(normal3)
		relu3 = Activation('relu')(conv3)
		pool3 = MaxPooling2D(pool_size=(1,2))(relu3)

		flat = Flatten()(pool3)

		drop1 = Dropout(0.5)(flat)

		dens1 = XnorDense(128,
						use_bias=False,
						activation='sigmoid', name='dens1')(drop1)
		drop2 = Dropout(0.5)(dens1)

		dens2 = XnorDense(self.nb_classes,
						  use_bias=False,
						  name='dens2')(drop2)
		# option to include temperature in softmax
		temp = 1.0
		temperature = Lambda(lambda x: x / temp)(dens2)
		last = Activation('softmax')(temperature)

		self.model = Model(input=inputs, output=last)

		adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		self.model.compile(
			loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])
		print (self.model.summary())
		return self

class ConvNNBinary(ConvNN):
	def setup(self,X_train_shape):
		print ('X_train shape', X_train_shape)

		# Input shape = (None,1,16,200)
		inputs = Input(shape=X_train_shape[1:])

		normal1 = BatchNormalization(axis=2, name='normal1')(inputs)
		conv1 = BinaryConv2D(
			16, kernel_size=(X_train_shape[2],5),
			padding='valid', strides=(1,2), use_bias=False,
			name='conv1')(normal1)
		relu1 = Activation('relu')(conv1)
		pool1 = MaxPooling2D(pool_size=(1,2))(relu1)

		normal2 = BatchNormalization(axis=1, name='normal2')(pool1)

		conv2 = BinaryConv2D(
			32, kernel_size=(1, 3),
			padding='valid', strides=(1,2), use_bias=False,
			name='conv2')(normal2)
		relu2 = Activation('relu')(conv2)
		pool2 = MaxPooling2D(pool_size=(1,2))(relu2)

		normal3 = BatchNormalization(axis=1, name='normal3')(pool2)

		conv3 = BinaryConv2D(
			64, kernel_size=(1, 3),
			padding='valid', strides=(1,1), use_bias=False,
			name='conv3')(normal3)
		relu3 = Activation('relu')(conv3)
		pool3 = MaxPooling2D(pool_size=(1,2))(relu3)

		flat = Flatten()(pool3)

		drop1 = Dropout(0.5)(flat)

		dens1 = BinaryDense(128, H=1,
						  use_bias=False,
						  activation='sigmoid', name='dens1')(drop1)
		drop2 = Dropout(0.5)(dens1)

		dens2 = BinaryDense(self.nb_classes, H=1, use_bias=False,
						  name='dens2')(drop2)
		# option to include temperature in softmax
		temp = 1.0
		temperature = Lambda(lambda x: x / temp)(dens2)
		last = Activation('softmax')(temperature)

		self.model = Model(input=inputs, output=last)

		adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		self.model.compile(
			loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])
		print (self.model.summary())
		return self

class ConvNNInt(ConvNN):
	def __init__(self,
				 target,batch_size=16,nb_classes=2,epochs=2,
				 mode='cv', model='int', bits=1):
		ConvNN.__init__(self,target,batch_size,nb_classes,epochs,mode,model)
		self.bits = bits
		print ('DEBUG: Using %d bits in Integer Layers' % self.bits)

	def setup(self,X_train_shape):
		print ('X_train shape', X_train_shape)

		# Input shape = (None,1,16,200)
		inputs = Input(shape=X_train_shape[1:])

		normal1 = BatchNormalization(axis=1, name='normal1')(inputs)
		conv1 = IntConv2D(
			16, kernel_size=(X_train_shape[2],5),
			bits=self.bits,
			padding='valid', strides=(1,2), use_bias=False,
			name='conv1')(normal1)
		relu1 = Activation('relu')(conv1)
		pool1 = MaxPooling2D(pool_size=(1,2))(relu1)

		normal2 = BatchNormalization(axis=1, name='normal2')(pool1)

		conv2 = IntConv2D(
			32, kernel_size=(1, 3),
			bits=self.bits,
			padding='valid', strides=(1,2), use_bias=False,
			name='conv2')(normal2)
		relu2 = Activation('relu')(conv2)
		pool2 = MaxPooling2D(pool_size=(1,2))(relu2)

		normal3 = BatchNormalization(axis=1, name='normal3')(pool2)

		conv3 = IntConv2D(
			64, kernel_size=(1, 3),
			bits=self.bits,
			padding='valid', strides=(1,1), use_bias=False,
			name='conv3')(normal3)
		relu3 = Activation('relu')(conv3)
		pool3 = MaxPooling2D(pool_size=(1,2))(relu3)

		flat = Flatten()(pool3)

		drop1 = Dropout(0.5)(flat)

		dens1 = IntDense(128,
						 bits=self.bits,
						  use_bias=False,
						  activation='sigmoid', name='dens1')(drop1)
		drop2 = Dropout(0.5)(dens1)

		dens2 = IntDense(self.nb_classes,
						 bits=self.bits,
						 use_bias=False,
						name='dens2')(drop2)
		# option to include temperature in softmax
		temp = 1.0
		temperature = Lambda(lambda x: x / temp)(dens2)
		last = Activation('softmax')(temperature)

		self.model = Model(input=inputs, output=last)

		adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		self.model.compile(
			loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])
		#print (self.model.summary())
		return self
