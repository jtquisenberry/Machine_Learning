# coding: utf-8
import nltk
from nltk.corpus import stopwords

def ImportStopwords():
    try:
        stops = set(stopwords.words("english"))
    except:
        nltk.download("stopwords")

def StripEmlHeaders(textToStrip):
# Given a string, return nothing if it is an email header,
# return the string if it is not an email header

    if textToStrip.startswith((u'From:',u'Date:',u'To:',u'Subject:',u'CC:', u'BCC:')):
        return u''
    else:
        return textToStrip

def StripPunctuation(textToStrip):
# Given a string, remove select punctuation

    puncs = u'!"#$%&\'()*+,-./:;<=>?[\]^_`{|}~‐‑‒–—‘’‚‛“”„‟†‡…'
    d = {ord(c): u' ' for c in puncs}
    newline = textToStrip.translate(d)
    return newline

def StripStopWords(textToStrip):
# Given a line, remove stop words
# returns a line, adds line break to the end because str.split removes all whitespace

    newline = u''
    stops = set(stopwords.words("english"))
    linewords = textToStrip.split()
    newwords = [word for word in linewords if word.lower() not in stops]
    for w in newwords:
        newline = newline + u' ' + w
    newline = newline + u'\r\n'
    return newline
