import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf
from random import randrange

def banner(msg):
    print(f'\n========\n{msg}\n--------')

def predict(model, row, actual, input_encoder_rasse, input_encoder_rasse2, input_encoder_geschlecht):
    row = row.to_numpy()
    row_rasse = row[0]
    row_geschlecht = row[1]
    row_rasse2 = row[2]
    print(f'Creating prediction for input:')
    print(f'Rasse: {input_encoder_rasse.inverse_transform([row_rasse])[0]} ({row_rasse})')
    print(f'Rasse 2: {input_encoder_rasse2.inverse_transform([row_rasse2])[0]} ({row_rasse2})')
    print(f'Geschlecht: {input_encoder_geschlecht.inverse_transform([row_geschlecht])[0]} ({row_geschlecht})')

    prediction = model(row.reshape(-1, 3)).numpy()
    print('Raw prediction:')
    print(prediction)
    print('Softmax prediction/probabilities:')
    print(tf.nn.softmax(prediction).numpy())
    predicted_class = np.argmax(prediction)
    print(f'Predicted class: {output_encoder.inverse_transform(np.array([[predicted_class]]))[0][0]} ({predicted_class})')
    print(f'Actual class: {(output_encoder.inverse_transform(np.array([actual])))[0][0]} ({actual[0]})')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print(f'Loss: {loss_fn([actual], prediction).numpy()}')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bsp3_tensorflow.py <input.csv>", file=sys.stderr)
        sys.exit(-1)

    start = time.perf_counter()

    # 1. Load dataset

    df = pd.read_csv(sys.argv[1])
    df = df[~df.ALTER.isnull()] # Can't predict something that isn't there
    end = time.perf_counter()
    banner(f'1. Loaded dataset in {end-start}s, transforming labels...')
    start = end

    # 2. Encode labels to numeric values

    input_encoder_rasse = LabelEncoder()
    input_encoder_geschlecht = LabelEncoder()
    input_encoder_rasse2 = LabelEncoder()
    output_encoder = OrdinalEncoder(dtype=np.int8)

    X_r = pd.DataFrame(
            data=input_encoder_rasse.fit_transform(df['RASSE1']),
            columns=['RASSE1']
            )
    X_g = pd.DataFrame(
            data=input_encoder_geschlecht.fit_transform(df['GESCHLECHT_HUND']),
            columns=['GESCHLECHT_HUND']
            )
    X_s = pd.DataFrame(
            data=input_encoder_rasse2.fit_transform(df['RASSE2']),
            columns=['RASSE2']
            )
    X = X_r.join([X_g, X_s])

    y = output_encoder.fit_transform(df['ALTER'].to_numpy().reshape(-1, 1))

    end = time.perf_counter()
    banner(f'2. Transformed labels to numeric values in {end-start}s, splitting...')
    start = end

    print(f'Transformed input features look like this:')
    print(X)
    #print('Transformed ordinal classes look like this:')
    #print(y)
    print('Encoded ordinal classes:')
    print(output_encoder.categories_)

    # 3. Split into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    end = time.perf_counter()
    banner(f'3. Split training/test set in {end-start}s ({X_train.shape[0]} in training, {X_test.shape[0]} in test set), creating model...')
    start = end

    # 4. Define and compile a Keras model
    n_features = len(X.columns)
    n_classes = df['ALTER'].unique().size

    print(f'Have {n_features} input features, {n_classes} classes for ALTER')
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)),
            tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(n_classes, activation='sigmoid')
            ])

    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['mse', 'accuracy'])

    end = time.perf_counter()
    banner(f'4. Created model in {end-start}s, making unfitted prediction (lol)...')
    start = end

    random_test_row = randrange(X_test.shape[0])
    print(f'Randomly selecting row {random_test_row} from the test set')
    row = X_test.iloc[random_test_row]
    actual = y_test[random_test_row]
    predict(model, row, actual,
            input_encoder_rasse, input_encoder_rasse2, input_encoder_geschlecht)


    # 5. Fit the model
    banner('Fitting the model, this will take a while...')
    model.fit(X_train, y_train, epochs=1000, batch_size=1024, verbose=0)
    end = time.perf_counter()
    banner(f'5. Fit the model in {end-start}s, evaluating...')
    start = end

    # 6. Evaluate the model
    loss, mse, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test MSE: {mse} Accuracy: {accuracy}')
    end = time.perf_counter()
    banner(f'6. Evaluated test accuracy in {end-start}s, making prediction...')
    start = end

    # 7. Create a prediction (again, but this time after fitting)
    predict(model, row, actual,
            input_encoder_rasse, input_encoder_rasse2, input_encoder_geschlecht)
