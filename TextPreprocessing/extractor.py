from zipfile import ZipFile
import shutil


class Extractor:
    def __init__(self, *args, **kwargs):
        pass

    def extract(self):
        pass

    def encode(self):
        pass


class ZipExtractor(Extractor):
    def __init__(self, source, destination):
        super().__init__()
        self.source = source
        self.destination = destination

    def extract(self):
        with ZipFile(self.source, 'r') as zip_file:
            zip_file.extractall(path=self.destination)




class TextExtractor(Extractor):
    def __init__(self, source, destination):
        super().__init__()
        self.source = source
        self.destination = destination

    def extract(self):
        shutil.copyfile(self.source, self.destination)


filetype_to_extractor_dict = {
    'text': TextExtractor,
    'zip': ZipExtractor
}