import os


class Encoder:
    def __init__(self, source, destination, source_encoding):
        self.source = source
        self.destination = destination
        self.source_encoding = source_encoding

    def encode(self):
        if os.path.isfile(self.source):
            with open(self.source, mode='rt', encoding=self.source_encoding) as f:
                text = f.read()
                with open(self.destination, mode='wt', encoding='utf-8') as f_out:
                    f.write(text)
        else:
            files = os.listdir(self.source)
            files = [f for f in files if f.endswith('.txt')]
            source_files = [os.path.join(self.source, f) for f in files]
            destination_files = [os.path.join(self.destination, f) for f in files]
            source_destination = zip(source_files, destination_files)
            for source, destination in source_destination:
                with open(source, mode='rt', encoding=self.source_encoding) as f:
                    text = f.read()
                    with open(destination, mode='wt', encoding='utf-8') as f_out:
                        f_out.write(text)

