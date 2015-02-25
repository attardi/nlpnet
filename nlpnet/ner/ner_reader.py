# -*- coding: utf-8 -*-

__author__ = 'Daniele Sartiano and Giuseppe Attardi'

"""
Class for dealing with NER data from CoNLL03 file.
"""

from ..reader import TaggerReader
import nlpnet.config as config

import sys

def toIOBES(sent):
    """Convert from IOB to IOBES notation."""
    l = len(sent)
    for i in range(l):
        tok = sent[i]
        if  i+1 == l or sent[i+1][1][0] != 'I':
            if tok[1][0] == 'B':
                tok[1] = 'S'+tok[1][1:]
            elif tok[1][0] == 'I':
                tok[1] = 'E'+tok[1][1:]
    return sent

def noGazetteer(words):
    return [0] * len(words)

def create_extractor(dict):
    def present(words):
        # check presence in dictionary possibly as multiword
        # set to 1 items corresponding to words present in dictionary
        res = [0] * len(words)
        for i, token in enumerate(words):
            entity = token.lower()
            if entity in dict:
                res[i] = 1
            for j in range(i+1, len(words)):
                entity += ' ' + words[j].lower()
                if entity in dict:
                    for k in range(i, j+1):
                        res[k] = 1
        return res
    return present

def gazetteer(file):
    """
    :return a map of feature extractors form dictionary file, one for each class type.
    A dictionary file consists of lines:
    TYPE WORD[ WORD]*
    """
    classes = {}
    for line in open(file):
        line = line.strip().decode('utf-8')
        c, words = line.split(None, 1)
        words = words.lower()
        if c not in classes:
            classes[c] = set()
        classes[c].add(words)

    extractors = {}
    for c,dict in classes.iteritems():
        extractors[c] = create_extractor(dict)

    return extractors

class NerReader(TaggerReader):
    """
    This class reads data from a CoNLL03 corpus and turns it into a format
    readable by the neural network for the NER tagging task.
    """
    
    def __init__(self, md=None, sentences=None, filename=None, load_dictionaries=True, variant=None):
        """
        :param sentences: a sequence of tagged sentences. Each sentence must be a 
            sequence of (token, tag) tuples. If None, the sentences are read from the 
            default location.
        """
        self.rare_tag = None
        self.tag_dict = {}      # tag IDs
        self.task = 'ner'

        # sets word_dict and tags_dict
        super(NerReader, self).__init__(md, load_dictionaries)

        if sentences:
            self.sentences = sentences
        else:
            self.sentences = []
           
            if filename:
                with open(filename, 'rb') as f:
                    sentence = []
                    for line in f:
                        line = line.strip()

                        if line:
                            (form, pos, iob) = unicode(line, 'utf-8').split()
                            sentence.append([form, iob])
                        else:
                            sentence = toIOBES(sentence)
                            self.sentences.append(sentence)
                            sentence = []

    def create_converter(self):
        """
        Sets up the token converter, which is responsible for transforming tokens into their
        feature vector indices
        """
        super(NerReader, self).create_converter()
        inGazetteer = gazetteer(self.md.paths[self.md.gazetteer])
        for c in self.md.gaz_classes:
            if c in inGazetteer:
                self.converter.add_extractor(inGazetteer[c])
            else:
                self.converter.add_extractor(noGazetteer)

class NerTagReader(NerReader):
    """
    This class reads data from a CoNLL03 corpus and turns it into a format
    readable by the neural network for the NER tagging task.
    """
    
    def __init__(self):
        """
        Read sentences from stdin.
        """
        self.task = 'ner'
        # loads word_dict and tags_dict
        self.load_dictionary()
        self.load_tag_dict()

        self.sentences = []
        sent = []
        for line in sys.stdin:
            line = line.decode('utf-8').strip()
            if line:
                (form, pos, tag) = line.split(None, 2)
                sent.append([form, pos])
            else:
                self.sentences.append(sent)
                sent = []

    def toIOB(self, tags):
        """Convert from IOBES to IOB notation."""
        for i in range(len(tags)):
            tag = tags[i]
            if tag[0] == 'S':
                tags[i] = 'B'+tag[1:]
            elif tag[0] == 'E':
                tags[i] = 'I'+tag[1:]
        return tags
