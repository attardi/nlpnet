# -*- coding: utf-8 -*-

"""
Class for dealing with POS data.
"""

from .. import utils
from ..reader import TaggerReader

class POSReader(TaggerReader):
    """
    This class reads data from a POS corpus and turns it into a format
    readable by the neural network for the POS tagging task.
    """
    
    def __init__(self, md=None, sentences=None, filename=None, load_dictionaries=True):
        """
        :param tagged_text: a sequence of tagged sentences. Each sentence must be a 
            sequence of (token, tag) tuples. If None, the sentences are read from the 
            default location.
        """
        self.task = 'pos'
        self.rare_tag = None
        super(POSReader, self).__init__(md, load_dictionaries)
        
        if sentences is not None:
            self.sentences = sentences
        else:
            self.sentences = []
            
            if filename is not None:
                with open(filename, 'rb') as f:
                    for line in f:
                        line = line.strip()
                        # Attardi: don't do it on tags, since PTB uses ``
                        #cleaned = utils.clean_text(unicode(line, 'utf-8'), False)
                        cleaned = unicode(line, 'utf-8')
                        #items = cleaned.split()
                        #self.sentences.append([item.split('_') for item in items])
                        # Attardi: '_' may appear in text
                        items = cleaned.split(' ')
                        self.sentences.append([item.split('\t') for item in items])
