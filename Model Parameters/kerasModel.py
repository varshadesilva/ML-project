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
    def whereToFindPath(self, pathToFile):
        filePath = os.path.join(pathToFile)
        with open(filePath, "r") as f:
            fileData = f.read()
        return fileData.split('\n')

    def changeFromSomethingToSomethingElse(self, x):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)
        return tokenizer.texts_to_sequences(x), tokenizer


    def convertLanguages(self, line):
        YList = {v: k for k, v in freTokens.word_index.items()}
        YList[0] = '<PAD>'
        line = [engTokens.word_index[each] for each in line.split()]
        line = pad_sequences([line], maxlen=preProcEng.shape[-1], padding='post')
        allSentencesList = np.array([line[0], preProcEng[0]])
        model = keras.models.load_model('Model Parameters/mymodel')
        preds = model.predict(allSentencesList, len(allSentencesList))
        return ' '.join([YList[np.argmax(x)] for x in preds[0]])


    def fillIn(self, data, size=None):
        return pad_sequences(data, maxlen=size, padding='post')


    def makeDataReadable(self, A, B):
        Xtemp, xTokens = self.changeFromSomethingToSomethingElse(A)
        Ytemp, yTokens = self.changeFromSomethingToSomethingElse(B)
        Xtemp = self.fillIn(Xtemp)
        Ytemp = self.fillIn(Ytemp)
        Ytemp = Ytemp.reshape(*Ytemp.shape, 1)
        return Xtemp, Ytemp, xTokens, yTokens

    def getWhatisRequired(self, listOfId, taglizer):
        wordsList = {id: word for word, id in taglizer.word_index.items()}
        wordsList[0] = '<PAD>'

        return ' '.join([wordsList[preds] for preds in np.argmax(listOfId, 1)])

    def howRUDoing(self, model):
        shape = (137861, 15)
        opLength = 21
        vocabSizeEng = 199
        vocabSizeFre = 344
        model = model(shape, opLength, vocabSizeEng, vocabSizeFre)

    def make(self, ipShape, opSeqLEn, vocabSizeEng, vocanSizeFre):
        lr = 0.003
        model = Sequential()
        model.add(Embedding(vocabSizeEng, 128, input_length=ipShape[1],
                            input_shape=ipShape[1:]))
        model.add(Bidirectional(GRU(128)))
        model.add(RepeatVector(opSeqLEn))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(TimeDistributed(Dense(512, activation='relu')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(vocanSizeFre, activation='softmax')))
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(lr),
                      metrics=['accuracy'])
        return model


m = Something()
trainEnglish = "Data/Train/english"
trainFrench = "Data/Train/french"
englishSentences = m.whereToFindPath(trainEnglish)
frenchSentences = m.whereToFindPath(trainFrench)
preProcEng, preProcFre, engTokens, freTokens = m.makeDataReadable(englishSentences, frenchSentences)
engSeqMaxLen = preProcEng.shape[1]
freSeqMaxLen = preProcFre.shape[1]
vocabSizeEng = len(engTokens.word_index)
vocanSizeFre = len(freTokens.word_index)

model = m.make(preProcEng.shape,
               preProcFre.shape[1],
               len(engTokens.word_index) + 1,
               len(freTokens.word_index) + 1)
model.summary()
model.fit(preProcEng, preProcFre, batch_size=1024, epochs=100, validation_split=0.2)
model.save('Model Parameters/mymodel')

