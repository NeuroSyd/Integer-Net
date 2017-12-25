from keras import backend as K

from keras.layers import Dense, Conv2D
from keras import constraints
from keras import initializers
from models.int_ops import integerize

class Clip(constraints.Constraint):
	def __init__(self, min_value, max_value=None):
		self.min_value = min_value
		self.max_value = max_value
		if not self.max_value:
			self.max_value = -self.min_value
		if self.min_value > self.max_value:
			self.min_value, self.max_value = self.max_value, self.min_value

	def __call__(self, p):
		return K.clip(p, self.min_value, self.max_value)

	def get_config(self):
		return {"name": self.__call__.__name__,
	            "min_value": self.min_value,
	            "max_value": self.max_value}

class IntDense(Dense):
	def __init__(self, units, bits=1, **kwargs):
		super(IntDense, self).__init__(units, **kwargs)
		self.bits = bits

	def build(self, input_shape):
		assert len(input_shape) >= 2
		input_dim = input_shape[1]

		self.kernel_constraint = Clip(-1, 1)
		self.kernel_initializer = initializers.RandomUniform(-1, 1)
		self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

		if self.use_bias:
			self.bias = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
		else:
			self.bias = None
		self.built = True

	def call(self, inputs):
		k_k, kernel_i = integerize(self.kernel, self.bits)
		k_i, inputs_i = integerize(inputs, self.bits)

		outputs = K.dot(inputs_i, kernel_i) * k_k * k_i

		if self.use_bias:
			k_b , bias_i = integerize(self.bias, self.bits)
			outputs = K.bias_add(outputs, bias_i * k_b)

		if self.activation is not None:
			return self.activation(outputs)  # pylint: disable=not-callable
		return outputs

class IntConv2D(Conv2D):
	def __init__(self, filters, bits=1, **kwargs):
		super(IntConv2D, self).__init__(filters, **kwargs)
		self.bits = bits


	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs '
		                         'should be defined. Found `None`.')

		input_dim = input_shape[channel_axis]
		#print ('DEBUG: input_shape', input_shape)
		kernel_shape = self.kernel_size + (input_dim, self.filters)
		#print ('DEBUG: kernel_shape', kernel_shape)

		self.kernel_constraint = Clip(-1, 1)
		self.kernel_initializer = initializers.RandomUniform(-1, 1)
		self.kernel = self.add_weight(shape=kernel_shape,
		                         initializer=self.kernel_initializer,
		                         name='kernel',
		                         regularizer=self.kernel_regularizer,
		                         constraint=self.kernel_constraint)

		if self.use_bias:
			self.bias = self.add_weight(shape=(self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

		else:
			self.bias = None

		self.built = True

	def call(self, inputs):
		_, kernel_i = integerize(self.kernel, self.bits)
		_, inputs_i = integerize(inputs, self.bits)

		if self.data_format == 'channels_last':
			data_format = 'NHWC'
			channel_axis = -1
		else:
			data_format = 'NCHW'
			channel_axis = 1

		outputs = K.conv2d(
			inputs_i, kernel_i,
			strides=self.strides,
			padding=self.padding,
			data_format=self.data_format
		)

		kernel_m = K.reshape(self.kernel, (-1, self.filters))
		kernel_k = K.stop_gradient(K.max(K.abs(kernel_m), axis=0)/2**self.bits)

		inputs_m = K.max(K.abs(inputs), axis=channel_axis, keepdims=True)/2**self.bits
		ones = K.ones(self.kernel_size + (1,1))
		inputs_k = K.conv2d(
			inputs_m, ones,
			strides=self.strides,
			padding=self.padding,
			data_format=self.data_format
		)

		if self.data_format == 'channels_first':
			outputs = outputs * \
					  K.stop_gradient(inputs_k) * \
					  K.expand_dims(K.expand_dims(K.expand_dims(kernel_k, 0), -1), -1)
		else:
			outputs = outputs * \
					  K.stop_gradient(inputs_k) * \
					  K.expand_dims(K.expand_dims(K.expand_dims(kernel_k, 0), 0), 0)

		if self.use_bias:
			outputs = K.bias_add(
				outputs,
				self.bias,
				data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)
		return outputs






	def get_config(self):
		config = {'bits': self.bits
		        }
		base_config = super(IntConv2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
