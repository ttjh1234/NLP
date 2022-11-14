import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

input_dims=32

def build_model(input_dims):
    # Input Layer
    input_layer = tf.keras.layers.Input(shape=(input_dims,))
    
    # Attention Layer
    attention_probs=tf.keras.layers.Dense(input_dims,activation='softmax')(input_layer)
    
    # Layer that multiplies (element-wise) a list of inputs.
    attention_mul = tf.keras.layers.multiply([input_layer,attention_probs])
    
    # FC Layer
    fc_attention_mul = tf.keras.layers.Dense(64)(attention_mul)
    y = tf.keras.layers.Dense(1,activation='sigmoid')(fc_attention_mul)\
    
    return tf.keras.Model(inputs=[input_layer],outputs=y)

model=build_model(input_dims)
model.summary()
tf.keras.utils.plot_model(model,show_shapes=True)

def get_data(n, input_dims,attention_column=1):
    train_x = np.random.standard_normal(size=(n,input_dims))
    train_y=np.random.randint(low=0,high=2, size=(n,1))
    train_x[:,attention_column]=train_y[:,0]
    
    return (train_x,train_y)

train_x, train_y = get_data(10000, 32, 5)
test_x,  test_y  = get_data(10000, 32, 5)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y,epochs=20,batch_size=64,validation_split=0.5,verbose=2)

# 모델의 중간층 값을 보고싶을 때, 입력에 대해 중간층의 출력을 반환하는 모델 생성

layer_outputs=[layer.output for layer in model.layers]
activation_model = tf.keras.Model(inputs=model.input,outputs=layer_outputs)

output_data = activation_model.predict(test_x)

model.summary()
activation_model.summary()

print(output_data[1], output_data[1].shape)

attention_vector = np.mean(output_data[1],axis=0)

print(attention_vector,attention_vector.shape)

df = pd.DataFrame(attention_vector.T,columns=['attention (%)'])

df.plot.bar()



