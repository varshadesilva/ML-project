from googletrans import Translator

def testTranslation(input):
    translator = Translator()
    result = translator.translate(input, dest='fr').text
    print(result)
