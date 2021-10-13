import sys
import time
import pandas as pd
from numpy import argmax, ravel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bsp3_tensorflow.py <input.csv>", file=sys.stderr)
        sys.exit(-1)

    start = time.perf_counter()

    # 1. Load dataset

    df = pd.read_csv(sys.argv[1])
    end = time.perf_counter()
    print(f'1. Loaded dataset in {end-start}s')
    start = end

    # 2. Encode labels to numeric values

    #input_encoder = OrdinalEncoder()
    input_encoder = LabelEncoder()
    output_encoder = LabelEncoder()

    #input_column = df['RASSE1'].to_numpy().reshape(-1, 1)
    X = pd.DataFrame(data=input_encoder.fit_transform(df['RASSE1']),
            columns=['RASSE1'])
    #print(X)
    y = output_encoder.fit_transform(df['ALTER'])
    #print(y)

    end = time.perf_counter()
    print(f'2. Transformed labels to numeric values in {end-start}s')
    start = end

    # 3. Split into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # 4. Define and compile a Keras model
    #
    # TODO: BEGIN POLISHING/RESTRUCTURING BASED ON ASSIGNMENT HERE
    #
    n_features = df['RASSE1'].unique().size
    n_classes = df['ALTER'].unique().size

    print(f'Have {n_features} different values for RASSE1, {n_classes} for ALTER')
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=[1]),
            tf.keras.layers.Dense(n_classes)
            ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    end = time.perf_counter()
    print(f'4. Created Keras model in {end-start}s')
    start = end

    # 5. Fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

    # 6. Evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print('Test Accuracy: %.3f' % acc)

    # 7. Create a prediction
    row = X.head(1)
    print(f'Creating prediction for input {input_encoder.inverse_transform(ravel(row.to_numpy()))} {row.values[0][0]}')
    prediction = model.predict([row])
    print(prediction)
    predicted_class = argmax(prediction)
    print(f'Predicted class: {(output_encoder.inverse_transform([predicted_class]))[0]} ({predicted_class})')
