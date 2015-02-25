# -*- coding: utf-8 -*-

import logging
import numpy as np
import re
from collections import Counter

import config
from word_dictionary import WordDictionary as WD
from collections import defaultdict

class Caps(object):
    """Dummy class for storing numeric values for capitalization."""
    # lower = 0
    # title = 1
    # non_alpha = 2
    # other = 3
    # upper = 4                   # Attardi
    # num_values = 5

    # SENNA
    padding = 0
    upper  = 1
    hascap = 2
    title  = 3
    nocaps = 4
    num_values = 5

class Token(object):
    def __init__(self, word, lemma='NA', pos='NA', morph='NA', chunk='NA'):
        """
        A token representation that stores discrete attributes to be given as 
        input to the neural network. 
        """
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.morph = morph
        self.chunk = chunk
    
    def __str__(self):
        return str(self.word)
    
    def __repr__(self):
        return self.word.__repr__()
    
class Affix(object):
    """Dummy class for manipulating suffixes/affixes and their related codes."""
    path = None                 # redefined in subclasses
    # words smaller than the suffix size
    #small_word = 0
    padding = 0
    other = 1                   # NOSUFFIX

    @classmethod
    def load(cls, md):
        """
        loads the listed suffixes/prefixes from the affix file.
        """
        cls.codes = {}
        code = cls.other + 1
        logger = logging.getLogger("Logger")
        try:
            with open(md.paths[cls.path], 'rb') as f:
                for line in f:
                    affix = unicode(line.strip(), 'utf-8')
                    cls.codes[affix] = code
                    code += 1
            #cls.count_size = len(affix)
        except IOError:
            logger.warning('Affix list doesn\'t exist.')
            raise
        cls.num_suffixes = code

class Suffix(Affix):
    path = 'suffixes'
    codes = {}
    # words smaller than the suffix size
    #small_word = 0
    padding = 0
    other = 1                   # NOSUFFIX
    
    # Attardi: fixed setting (mimic SENNA)
    count = 2

    @classmethod
    def create_suffix_list(cls, wordlist, num, size, min_occurrences):
        """
        Creates a file containing the list of the most common suffixes found in 
        wordlist.
        
        :param wordlist: a list of word types (there shouldn't be repetitions)
        :param num: maximum number of suffixes
        :param size: desired size of suffixes
        :param min_occurrences: minimum number of occurrences of each suffix
        in wordlist
        """
        all_endings = [x[-size:] for x in wordlist 
                       if len(x) > size
                       and not re.search('_|\d', x[-size:])]
        c = Counter(all_endings)
        common_endings = c.most_common(num)
        suffix_list = [e for e, n in common_endings if n >= min_occurrences]
        
        with open(config.FILES[cls.path], 'wb') as f:
            for suffix in suffix_list:
                f.write('%s\n' % suffix.encode('utf-8'))

    @classmethod
    def get(cls, word):
        """
        Returns the suffix code for the given word.
        """
        # if len(word) < Suffix.suffix_size: return Suffix.small_word
        
        # Attardi: handle padding
        if word == WD.padding_left or word == WD.padding_left:
            return Suffix.padding

        suffix = word[-Suffix.count:]

        return Suffix.codes.get(suffix.lower(), Suffix.other)

    @classmethod
    def get_all(cls, words):
        """
        :return: the list of suffix codes for the given words.
        """
        return map(cls.get, words)

class Prefix(Affix):
    path = 'prefixes'
    codes = {}
    num_affixes = 2
    
    # Attardi: fixed setting (mimic SENNA)
    affix_size = 2

class TokenConverter(object):
    
    def __init__(self):
        """
        Class to convert tokens into indices to their feature vectos in
        feature matrices.
        """
        self.extractors = []
    
    def add_extractor(self, extractor):
        """
        Adds an extractor function to the TokenConverter. In order to get a token's 
        feature indices, the Converter will call each of its extraction functions passing
        the token as a parameter. The result will be a list containing each result. 
        """
        self.extractors.append(extractor)
    
    def get_padding_left(self, tokens_as_string=True):
        """
        Returns an object to be used as the left padding in the sentence.
        
        :param tokens_as_string: if True, treat tokens as strings. 
        If False, treat them as Token objects.
        """
        if tokens_as_string:
            pad = WD.padding_left
        else:
            pad = Token(WD.padding_left)
        return self.convert([pad])[0]
    
    def get_padding_right(self, tokens_as_string=True):
        """
        Returns an object to be used as the right padding in the sentence.
        
        :param tokens_as_string: if True, treat tokens as strings. 
            If False, treat them as Token objects.
        """
        if tokens_as_string:
            pad = WD.padding_right
        else:
            pad = Token(WD.padding_right)
        return self.convert([pad])[0]
    
    def convert(self, sent):
        """
        Converts a sentence into a 2-d array of feature indices.
        """
        indices = np.array(zip(*[extractor(sent) for extractor in self.extractors]))
        return indices

# capitalization
def get_capitalization(word):
    """
    Returns a code describing the capitalization of the word:
    lower, title, upper, other or non-alpha (numbers and other tokens that can't be
    capitalized).
    """
    # if not word.isalpha():
    #     return Caps.non_alpha
    
    # if word.istitle():
    #     return Caps.title
    
    # if word.islower():
    #     return Caps.lower

    # if word.isupper():          # Attardi
    #     return Caps.upper
    
    # return Caps.other

    # SENNA
    if word == WD.padding_left or word == WD.padding_right:
        return Caps.padding

    if word.isupper():
        return Caps.upper

    if word[0].isupper():       # istitle() checks other letters too
        return Caps.title
    
    # can't use islower() because it accepts '3b'
    for c in word:
        if c.isupper():
            return Caps.hascap

    return Caps.nocaps

# capitalization
def get_capitalizations(words):
    """
    :return: the list of capitalization codes for the given list of words.
    """
    return map(get_capitalization, words)

def capitalize(word, capitalization):
    """
    Capitalizes the word in the desired format. If the capitalization is 
    Caps.other, it is set all uppercase.
    """
    if capitalization == Caps.non_alpha:
        return word
    elif capitalization == Caps.lower:
        return word.lower()
    elif capitalization == Caps.title:
        return word.title()
    elif capitalization == Caps.other:
        return word.upper()
    else:
        raise ValueError("Unknown capitalization type.")

# Gazetteer feature is binary (0, 1)
num_gazetteer_tags = 2


