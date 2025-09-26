from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization, Bidirectional
from keras.regularizers import l2
from keras.optimizers import Adam
from constants import LENGTH_KEYPOINTS

def get_model(max_length_frames, output_length: int):
    model = Sequential()
    # >>> Ignora correctamente los ceros del padding
    model.add(Masking(mask_value=0.0, input_shape=(max_length_frames, LENGTH_KEYPOINTS)))

    # >>> Menos regularización y BiLSTM para captar mejor contexto
    model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(96, return_sequences=False, kernel_regularizer=l2(1e-4))))
    model.add(Dropout(0.3))

    # >>> Normalización ayuda a la estabilidad
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dense(output_length, activation='softmax'))

    # >>> LR explícito y más bajo si tus features son ruidosos
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
