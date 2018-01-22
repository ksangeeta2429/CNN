# CNN
This project is basically for evaluation of Convolutional Neural Nets. I train the models with Keras (in Python) but since the model is
to be run on an embedded device without GPU support/Deep Learning libraries, the Forward Pass (required for classifying a data) is written
from scratch in C#.

Currently the code is hardcoded with my model's architecture which is mentioned below:

Architecture as in Keras:

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 8), strides=(1, 4), input_shape=(16, 32, 1), padding='same', activation='relu', kernel_constraint=maxnorm(3), data_format='channels_last'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(1, 3), strides=(1, 1), activation='relu', padding='same', kernel_constraint=maxnorm(3)))\
model.add(MaxPooling2D(pool_size=a(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

The Weights of the above trained model is dumped into a json file in the following way:\
```python
dict = {}
dict["conv1"]       = model.layers[0].get_weights()[0].tolist()     # Shape: (1, 8, 1, 32)
dict["bias_conv1"]  = model.layers[0].get_weights()[1].tolist()     # Shape: (32,)
dict["conv2"]       = model.layers[2].get_weights()[0].tolist()     # Shape: (1, 3, 32, 32)
dict["bias_conv2"]  = model.layers[2].get_weights()[1].tolist()     # Shape: (32,)
dict["dense1"]      = model.layers[5].get_weights()[0].tolist()     # Shape: (1024, 512)
dict["bias_dense1"] = model.layers[5].get_weights()[1].tolist()     # Shape: (512,)
dict["dense2"]      = model.layers[7].get_weights()[0].tolist()     # Shape: (512, 1)
dict["bias_dense2"] = model.layers[7].get_weights()[1].tolist()     # Shape: (1,)
dict["last"] = model.get_weights()[7].tolist() Last Layer Weight 

import json
with open('weights.json', 'w') as fp:
    json.dump(dict, fp)
```

Before running the model, the following two things are done:
1. Weights are loaded in desired data format by deserializing JSON using Newtonsoft.Json with readJSON() 
2. Test Data (to be classified) is loaded from Excel and converted into the desired format with getDataTensor()

File to begin with: **CNN.cs**
