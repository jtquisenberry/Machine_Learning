import requests
import os
from corpus_settings import corpus_definitions
from extractor import *
from encoder import *

CORPORA_PATH = './corpora'


class Corpus:
    def __init__(self, definition: dict):
        self.filename = definition.get('filename', None)
        self.url = definition.get('url', None)
        self.subdirectory = definition.get('subdirectory', None)
        self.filetype = definition.get('filetype', None)
        self.source_encoding = definition.get('source_encoding', None)

    def setup(self):
        self.download()
        self.extract()
        self.encode_as_utf8()

    def download(self, force=False):
        url = self.url
        subdirectory = self.subdirectory
        filename = self.filename
        destination_path = os.path.join(CORPORA_PATH, subdirectory)
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)
        destination = os.path.join(destination_path, filename)
        if force or not os.path.exists(destination):
            r = requests.get(url)
            with open(destination, 'wb') as f:
                f.write(r.content)
        return True

    def extract(self):
        extractor_class: type = filetype_to_extractor_dict[self.filetype]
        source_path = os.path.join('./corpora', self.subdirectory)
        destination_path = os.path.join('./corpora_extracted', self.subdirectory)
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)
        source = os.path.join(source_path, self.filename)
        if self.filetype == 'text':
            destination = os.path.join(destination_path, self.filename)
        else:
            destination = destination_path
        extractor = extractor_class(source, destination)
        extractor.extract()

    def encode_as_utf8(self):
        source = os.path.join('./corpora_extracted', self.subdirectory)
        destination = os.path.join('./corpora_encoded', self.subdirectory)
        if not os.path.exists(destination):
            os.mkdir(destination)
        encoder = Encoder(source, destination, source_encoding=self.source_encoding)
        encoder.encode()




if __name__ == '__main__':
    corpus_name = 'princess_bride'
    corpus_definition = corpus_definitions[corpus_name]
    corpus = Corpus(corpus_definition)
    corpus.setup()


