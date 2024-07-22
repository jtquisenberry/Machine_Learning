import text_process
import os
import codecs

text_process.ImportStopwords()

inpath = "E:\ML\Corpora\spambot\headers stripped"
for root,dirs,filenames in os.walk(inpath):
    for f in filenames:
        outfile = os.path.join("E:\ML\Corpora\spambot\hpsl",f).replace(".txt","_out.txt")
        with codecs.open(outfile, 'w',encoding='utf-8') as ow:
            with codecs.open(os.path.join(root,f), 'r',encoding='utf-8') as infile:
                line = u''
                for line in infile.readlines():
                    newline = u''
                    newline = newline + text_process.StripEmlHeaders(line)
                    if newline.strip() != '':
                        newline = text_process.StripPunctuation(newline)
                    if newline.strip() != '':
                        newline = text_process.StripStopWords(newline)
                    if newline.strip() != '':
                        ow.write(newline.lower())
