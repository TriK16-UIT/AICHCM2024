import googletrans

class Translation():
    def __init__(self, from_lang="vi", to_lang="en"):
        self.__to_lang = to_lang
        self.__from_lang = from_lang
        self.translator = googletrans.Translator()

    def translate(self, text):
        return self.translator.translate(text, src = self.__from_lang, dest=self.__to_lang).text
    
