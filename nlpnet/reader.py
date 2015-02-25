#!/usr/env python
# -*- coding: utf-8 -*-

"""
Base class for reading NLP tagging data.
"""

import os
import logging
import numpy as np
from collections import Counter

import attributes
import metadata
import config
from word_dictionary import WordDictionary, NgramDictionary
from attributes import get_capitalizations, Prefix, Suffix


class TextReader(object):
    
    def __init__(self, md=None, sentences=None, filename=None, variant=None):
        """
        :param sentences: A list of lists of tokens.
        :param filename: Alternatively, the name of the file from where sentences 
            can be read. The file should have one sentence per line, with tokens
            separated by white spaces.
        """
        self.variant = variant
        if sentences is not None:
            self.sentences = sentences
        else:
            self.sentences = []
            with open(filename, 'rb') as f:
                for line in f:
                    sentence = unicode(line, 'utf-8').split()
                    self.sentences.append(sentence)
                    
        self.converter = None
        self.task = 'lm'
        self._set_metadata(md)
    
    def _set_metadata(self, md):
        if md is None:
            #metadata not provided = using global data_dir for files
            self.md = metadata.Metadata(self.task, config.FILES)
        else:
            self.md = md

    def add_text(self, text):
        """
        Adds more text to the reader. The text must be a sequence of sequences of 
        tokens.
        """
        self.sentences.extend(text)
    
    def get_dictionaries(self, dict_size=None):
        self.load_or_create_dictionary(dict_size)
        self.load_or_create_tag_dict()

    def load_or_create_dictionary(self, dict_size=None):
        """
        Try to load the vocabulary from the default location. If the vocabulary
        file is not available, create a new one from the sentences available
        and save it.
        """
        if os.path.isfile(self.md.paths['vocabulary']):
            self.load_dictionary()
            return
        
        self.generate_dictionary(dict_size, minimum_occurrences=2)
        self.save_dictionary()
    
    def load_or_create_tag_dict(self):
        """No tag dictioinary"""
        return

    def load_dictionary(self):
        """Read a file with a word list and create a dictionary."""
        logger = logging.getLogger("Logger")
        logger.info("Loading vocabulary")
        filename = self.md.paths['vocabulary']
        
        words = []
        with open(filename, 'rb') as f:
            for word in f:
                word = unicode(word, 'utf-8').strip()
                if word:
                    words.append(word)
        
        wd = WordDictionary.init_from_wordlist(words)
        self.word_dict = wd
        logger.info("Done. Dictionary size is %d types" % wd.num_tokens)
    
    def generate_dictionary(self, dict_size=None, minimum_occurrences=2):
        """
        Generates a token dictionary based on the supplied text.
        
        :param dict_size: Max number of tokens to be included in the dictionary.
        :param minimum_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        """
        logger = logging.getLogger("Logger")
        logger.info("Creating dictionary...")
        
        self.word_dict = WordDictionary(self.sentences, dict_size,
                                        minimum_occurrences, variant=self.variant)
            
        logger.info("Done. Dictionary size is %d tokens" % self.word_dict.num_tokens)
    
    def save_dictionary(self, filename=None):
        """
        Saves the reader's word dictionary as a list of words.
        
        :param filename: path to the file to save the dictionary. 
            if not given, it will be saved in the default nlpnet
            data directory.
        """
        logger = logging.getLogger("Logger")
        if filename is None:
            filename = self.md.paths['vocabulary']
        
        self.word_dict.save(filename)
        logger.info("Dictionary saved in %s" % filename)
    
    def codify_sentences(self):
        """
        Converts each token in each sequence into indices to their feature vectors
        in feature matrices. The previous sentences as text are not accessible anymore.
        """
        new_sentences = []
        for sent in self.sentences:
            new_sent = self.converter.convert(sent)
            new_sentences.append(new_sent)
        
        self.sentences = new_sentences
    
    def create_suffix_list(self, max_size, min_occurrences):
        """
        Check if there exists a suffix list in the data directory. If there isn't,
        create a new one based on the training sentences.
        """
        if os.path.isfile(self.md.paths['suffixes']):
            return
        
        logger = logging.getLogger("Logger")
        suffixes_all_lengths = []
        # only get the suffix size n from words with length at least (n+1)
        types = {token.lower() for sent in self.sentences for token, _ in sent}
        for length in range(1, max_size + 1):
            c = Counter(type_[-length:]
                        for type_ in types
                        if len(type_) > length)
            suffixes_this_length = [suffix for suffix in c 
                                    if c[suffix] >= min_occurrences]
            suffixes_all_lengths.extend(suffixes_this_length)
        
        logger.info('Created a list of %d sufixes.' % len(suffixes_all_lengths))
        text = '\n'.join(suffixes_all_lengths)
        with open(self.md.paths['suffixes'], 'wb') as f:
            f.write(text.encode('utf-8'))
    
    def create_prefix_list(self, max_size, min_occurrences):
        """
        Check if there exists a prefix list in the data directory. If there isn't,
        create a new one based on the training sentences.
        """
        if os.path.isfile(self.md.paths['prefixes']):
            return
        
        logger = logging.getLogger("Logger")
        prefixes_all_lengths = []
        # only get the prefix size n from words with length at least (n+1)
        types = {token.lower() for sent in self.sentences for token, _ in sent}
        for length in range(1, max_size + 1):
            c = Counter(type_[:length]
                        for type_ in types
                        if len(type_) > length)
            prefixes_this_length = [prefix for prefix in c 
                                    if c[prefix] >= min_occurrences]
            prefixes_all_lengths.extend(prefixes_this_length)
        
        logger.info('Created a list of %d prefixes.' % len(prefixes_all_lengths))
        text = '\n'.join(prefixes_all_lengths)
        with open(self.md.paths['prefixes'], 'wb') as f:
            f.write(text.encode('utf-8'))
    
    def create_converter(self):
        """
        Sets up the token converter, which is responsible for transforming tokens into their
        feature vector indices
        """
        def add_affix_extractors(affix):
            """
            Helper function that works for both suffixes and prefixes.
            :parame affix: either Suffix or Prefix.
            """
            loader_function = getattr(affix, 'load')
            loader_function(self.md)
            
            getter = getattr(affix, 'get_all')
            self.converter.add_extractor(getter)

        self.converter = attributes.TokenConverter()
        self.converter.add_extractor(self.word_dict.get_indices)
        if self.md.use_caps:
            self.converter.add_extractor(get_capitalizations)
        if self.md.use_prefix:
            add_affix_extractors(Prefix)
        if self.md.use_suffix:
            add_affix_extractors(Suffix)


class TaggerReader(TextReader):
    """
    Abstract class extending TextReader with useful functions
    for tagging tasks. 
    """
    
    def __init__(self, md=None, load_dictionaries=True):
        '''
        This class shouldn't be used directly. The constructor only
        provides method calls for subclasses.
        '''
        self._set_metadata(md)
        self.codified = False

        if load_dictionaries:
            self.load_dictionary()
            self.load_tag_dict()
    
    def load_or_create_tag_dict(self):
        """
        Try to load the tag dictionary from the default location. If the dictinaty
        file is not available, scan the available sentences and create a new one. 
        """
        key = '%s_tag_dict' % self.task
        filename = self.md.paths[key]
        if os.path.isfile(filename):
            self.load_tag_dict(filename)
            return
        
        tags = {tag for sent in self.sentences for _, tag in sent}
        self.tag_dict = {tag: code for code, tag in enumerate(tags)}
        self.save_tag_dict(filename)

    def generate_dictionary(self, dict_size=None, minimum_occurrences=2):
        """
        Generates a token dictionary based on the given sentences.
        
        :param dict_size: Max number of tokens to be included in the dictionary.
        :param minimum_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        """
        logger = logging.getLogger("Logger")
        logger.info("Creating dictionary...")
        
        tokens = [token for sent in self.sentences for token, _ in sent]
        self.word_dict = WordDictionary(tokens, dict_size, minimum_occurrences,
                                        variant=self.variant)
            
        logger.info("Done. Dictionary size is %d tokens" % self.word_dict.num_tokens)

    def get_inverse_tag_dictionary(self):
        """
        Returns a version of the tag dictionary that maps numbers to tags.
        Used for consulting the meaning of the network's output.
        """
        tuples = [(x[1], x[0]) for x in self.tag_dict.iteritems()]
        ret = dict(tuples)
        
        return ret
    
    def codify_sentences(self):
        """
        Converts each token in each sequence into indices to their feature vectors
        in feature matrices. The previous sentences as text are not accessible anymore.
        """
        new_sentences = []
        self.tags = []
        rare_tag_value = self.tag_dict.get(self.rare_tag)
        
        for sent in self.sentences:
            sentence_tags = []
            
            new_sent = self.converter.convert([token for token, tag in sent])
            sentence_tags = [self.tag_dict.get(tag, rare_tag_value) for token, tag in sent]
            new_sentences.append(new_sent)
            self.tags.append(np.array(sentence_tags))
        
        self.sentences = new_sentences
        self.codified = True
    
    def get_word_counter(self):
        """
        Returns a Counter object with word type occurrences.
        """
        c = Counter(token.lower() for sent in self.sentences for token, _ in sent)
        return c
    
    def get_tag_counter(self):
        """
        Returns a Counter object with tag occurrences.
        """
        return Counter(tag for sent in self.sentences for _, tag in sent)
    
    def save_tag_dict(self, filename=None, tag_dict=None):
        """
        Saves the tag dictionary to a file as a list of tags.
        
        :param tag_dict: the dictionary to save. If None, the default
            tag_dict for the class will be saved.
        :param filename: path to the file to save the dictionary. 
            If None, the class default tag_dict filename will be used.
        """
        if tag_dict is None:
            tag_dict = self.tag_dict
        if filename is None:
            key = '%s_tag_dict' % self.task
            filename = self.md.paths[key]

        ordered_keys = sorted(tag_dict, key=tag_dict.get)
        text = '\n'.join(ordered_keys)
        with open(filename, 'wb') as f:
            f.write(text.encode('utf-8'))
    
    def load_tag_dict(self, filename=None):
        """
        Load the tag dictionary from the default file and assign
        it to the tag_dict attribute. 
        """
        if filename is None:
            key = '%s_tag_dict' % self.task
            filename = self.md.paths[key]
            
        self.tag_dict = {}
        with open(filename, 'rb') as f:
            code = 0
            for tag in f:
                tag = unicode(tag, 'utf-8').strip()
                if tag:
                    self.tag_dict[tag] = code
                    code += 1
    
class TweetReader(TextReader):
    """
    Reader for tweets in SemEval 2013 format, one tweet per line consisting  of:
    SID	UID	polarity	tokenized text
    264183816548130816      15140428        positive      Gas by my house hit $3.39!!!! I'm going to Chapel Hill on Sat. :)
    """
    polarity_field = 2
    text_field = 3

    def __init__(self, md=None, ngrams=1, filename=None, variant=None):
        """
	:param ngrams: the lenght of ngrams to consider
        :param filename: the name of the file containing tweets. The file should have one tweet per line.
	:param variant: whether to use native, or SENNA or Polyglot conventions
        """
	self.ngrams = ngrams
        self.variant = variant
        self.sentences = []
        self.polarities = []
        with open(filename, 'rb') as f:
            for line in f:
                tweet = unicode(line, 'utf-8').split('\t')
                if tweet[TweetReader.polarity_field] == 'positive':
                    polarity = 1
                if tweet[TweetReader.polarity_field] == 'negative':
                    polarity = -1
                else:
                    continue
                self.sentences.append(tweet[TweetReader.text_field].split())
                self.polarities.append(polarity)
                    
        self.converter = None
        self.task = 'sslm'
        self._set_metadata(md)
    
    def generate_dictionary(self, dict_size=None, minimum_occurrences=None):
        """
        Generates a dictionary of all ngrams from the given sentences.
        
        :param dict_size: Max number of tokens to be included in the dictionary.
        :param minimum_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        """
        logger = logging.getLogger("Logger")
        logger.info("Creating dictionary...")
        
        # unigrams
        ngrams = [[token for sent in self.sentences for token in sent]]
        # multigrams
	for n in xrange(2, self.ngrams + 1):
            ngrams.append([])
	    for sent in self.sentences:
	    	for i in xrange(len(sent) + 1 - n):
		    ngrams[n-1].append(' '.join(sent[i:i+n]))
        self.word_dict = NgramDictionary(ngrams, dict_size, minimum_occurrences, variant=self.variant)

        logger.info("Done. Dictionary size is %d tokens" % self.word_dict.num_tokens)
