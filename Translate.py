from googletrans import Translator
import metrics
def testTranslation(input):
    translator = Translator()
    result = translator.translate(input, dest='fr').text
    print(result)

