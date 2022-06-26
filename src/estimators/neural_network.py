from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier


class NeuralNetworkClassifier(KerasClassifier):
    def __init__(self, initializer: str = 'glorot_uniform', optimizer: str = 'adam', **kc_kwargs):
        self.initializer = initializer
        self.optimizer = optimizer
        kc_kwargs['build_fn'] = self._build_model_experimental()
        super().__init__(**kc_kwargs)

    def _build_model(self):
        # create model
        model = Sequential()
        model.add(Dense(11, input_dim=10, kernel_initializer=self.initializer, activation='relu'))
        model.add(Dense(8, kernel_initializer=self.initializer, activation='relu'))
        model.add(Dense(3, kernel_initializer=self.initializer, activation='softmax'))
        # Compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def _build_model_experimental(self):
        # create model
        print("Experimental")
        model = Sequential()
        model.add(Dense(512, input_dim=10, kernel_initializer=self.initializer, activation='relu'))
        model.add(Dense(256, kernel_initializer=self.initializer, activation='relu'))
        model.add(Dense(128, kernel_initializer=self.initializer, activation='relu'))
        model.add(Dense(64, kernel_initializer=self.initializer, activation='relu'))
        model.add(Dense(32, kernel_initializer=self.initializer, activation='relu'))
        model.add(Dense(3, kernel_initializer=self.initializer, activation='softmax'))
        # Compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model
