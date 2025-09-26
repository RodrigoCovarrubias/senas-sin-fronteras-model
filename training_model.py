# Cambios clave marcados con >>> 
import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
from helpers import get_word_ids, get_sequences_and_labels
from constants import *

def training_model(model_path, epochs=300, batch_size=16, seed=42):
    word_ids = get_word_ids(WORDS_JSON_PATH)
    sequences, labels = get_sequences_and_labels(word_ids)

    # >>> usa float32 (más estable) y padding POST si usarás Masking en 0.0
    sequences = pad_sequences(
        sequences,
        maxlen=int(MODEL_FRAMES),
        padding='post', truncating='post', dtype='float32'
    )

    X = np.array(sequences, dtype='float32')
    y_int = np.array(labels, dtype='int32')
    y = to_categorical(y_int)  # deja float

    # >>> split más robusto y estratificado por clase
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y_int
    )

    # model = get_model(int(MODEL_FRAMES), len(word_ids))
    model = load_model("models/actions_15.h5", compile=True)

    # >>> callbacks: valida por val_loss y reduce LR si se estanca
    cbs = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        shuffle=True
    )

    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
