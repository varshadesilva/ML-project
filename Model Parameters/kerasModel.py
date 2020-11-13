import collections
import numpy as np
import os
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.utils import to_categorical

class Something:
    def whereToFindPath(self, path):
        file = os.path.join(path)
        with open(file, "r") as f:
            data = f.read()
        return data.split('\n')

    def changeFromSomethingToSomethingElse(self, x):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)
        return tokenizer.texts_to_sequences(x), tokenizer


    def convertLanguages(self, sentence):
        y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
        y_id_to_word[0] = '<PAD>'
        sentence = [english_tokenizer.word_index[word] for word in sentence.split()]
        sentence = pad_sequences([sentence], maxlen=preproc_english_sentences.shape[-1], padding='post')
        sentences = np.array([sentence[0], preproc_english_sentences[0]])
        model = keras.models.load_model('Model Parameters/mymodel')
        predictions = model.predict(sentences, len(sentences))
        return ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])


    def fillIn(self, x, length=None):
        return pad_sequences(x, maxlen=length, padding='post')


    def makeDataReadable(self, x, y):
        preprocess_x, x_tk = self.changeFromSomethingToSomethingElse(x)
        preprocess_y, y_tk = self.changeFromSomethingToSomethingElse(y)
        preprocess_x = self.fillIn(preprocess_x)
        preprocess_y = self.fillIn(preprocess_y)
        preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
        return preprocess_x, preprocess_y, x_tk, y_tk

    def getWhatisRequired(self, idList, tokenizer):
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'

        return ' '.join([index_to_words[prediction] for prediction in np.argmax(idList, 1)])

    def howRUDoing(self, model_final):
        input_shape = (137861, 15)
        output_sequence_length = 21
        english_vocab_size = 199
        french_vocab_size = 344
        model = model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)

    def make(self, input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
        learning_rate = 0.003
        model = Sequential()
        model.add(Embedding(english_vocab_size, 128, input_length=input_shape[1],
                            input_shape=input_shape[1:]))
        model.add(Bidirectional(GRU(128)))
        model.add(RepeatVector(output_sequence_length))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(TimeDistributed(Dense(512, activation='relu')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy'])
        return model


m = Something()
trainEnglish = "Data/Train/english"
trainFrench = "Data/Train/french"
englishSentences = m.whereToFindPath(trainEnglish)
frenchSentences = m.whereToFindPath(trainFrench)
preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = m.makeDataReadable(englishSentences, frenchSentences)
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

model = m.make(preproc_english_sentences.shape,
               preproc_french_sentences.shape[1],
               len(english_tokenizer.word_index) + 1,
               len(french_tokenizer.word_index) + 1)
model.summary()
model.fit(preproc_english_sentences, preproc_french_sentences, batch_size=1024, epochs=100, validation_split=0.2)
model.save('Model Parameters/mymodel')

