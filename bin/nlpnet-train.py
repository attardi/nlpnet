#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train a neural network for NLP tagging tasks.

Author: Erick Rocha Fonseca
"""

import logging
import numpy as np

# Attardi: allow executing from anywhere without installing the package
import sys
import os
import distutils.util
builddir = os.path.dirname(os.path.realpath(__file__)) + '/../build/lib.'
libdir = builddir + distutils.util.get_platform() + '-' + '.'.join(map(str, sys.version_info[:2]))
sys.path.append(libdir)

import nlpnet.config as config
import nlpnet.read_data as read_data
import nlpnet.utils as utils
import nlpnet.taggers as taggers
import nlpnet.metadata as metadata
import nlpnet.srl as srl
import nlpnet.pos as pos
import nlpnet.ner as ner
import nlpnet.arguments as arguments
import nlpnet.reader as reader
import nlpnet.attributes as attributes
from nlpnet.network import Network, ConvolutionalNetwork, LanguageModel, SentimentModel


############################
### FUNCTION DEFINITIONS ###
############################

def create_reader(args, md):
    """
    Creates and returns a TextReader object according to the task at hand.
    """
    logger.info("Reading text...")
    if args.task == 'pos':
        text_reader = pos.pos_reader.POSReader(md, filename=args.gold, variant=args.variant)
        if args.suffix:
            text_reader.create_suffix_list(args.suffix_size, 5)
        if args.prefix:
            text_reader.create_prefix_list(args.prefix_size, 5)

    elif args.task == 'ner':
        text_reader = ner.ner_reader.NerReader(md, filename=args.gold,
                                               variant=args.variant)

    elif args.task == 'lm':
        text_reader = reader.TextReader(md, filename=args.gold,
                                        variant=args.variant)
        text_reader.get_dictionaries(args.dict_size_size)

    elif args.task == 'sslm':
        text_reader = reader.TweetReader(md, filename=args.gold,
                                         ngrams=args.ngrams, variant=args.variant)
        text_reader.get_dictionaries(args.dict_size_size)

    elif args.task.startswith('srl'):
        text_reader = srl.srl_reader.SRLReader(md, filename=args.gold, only_boundaries=args.identify, 
                                               only_classify=args.classify,
                                               only_predicates=args.predicates,
                                               variant=args.variant)
    
        if args.identify:
            # only identify arguments
            text_reader.convert_tags('iobes', only_boundaries=True)
            
        elif not args.classify and not args.predicates:
            # this is SRL as one step, we use IOB
            text_reader.convert_tags('iob', update_tag_dict=False)
        
    else:
        raise ValueError("Unknown task: %s" % args.task)
    
    return text_reader
    

def create_network(args, text_reader, feature_tables, md=None):
    """Creates and returns the neural network according to the task at hand."""
    logger = logging.getLogger("Logger")

    if args.task.startswith('srl') and args.task != 'srl_predicates':
        num_tags = len(text_reader.tag_dict)
        distance_tables = utils.set_distance_features(args.max_dist, args.target_features,
                                                      args.pred_features)
        nn = ConvolutionalNetwork.create_new(feature_tables, distance_tables[0], 
                                             distance_tables[1], args.window, 
                                             args.convolution, args.hidden, num_tags)
        padding_left = text_reader.converter.get_padding_left(False)
        padding_right = text_reader.converter.get_padding_right(False)
        if args.identify:
            logger.info("Loading initial transition scores table for argument identification")
            transitions = srl.train_srl.init_transitions_simplified(text_reader.tag_dict)
            nn.transitions = transitions
            nn.learning_rate_trans = args.learning_rate_transitions
            
        elif not args.classify:
            logger.info("Loading initial IOB transition scores table")
            transitions = srl.train_srl.init_transitions(text_reader.tag_dict, 'iob')
            nn.transitions = transitions
            nn.learning_rate_trans = args.learning_rate_transitions
    
    elif args.task == 'lm':
        nn = LanguageModel.create_new(feature_tables, args.window, args.hidden)
        padding_left = text_reader.converter.get_padding_left(tokens_as_string=True)
        padding_right = text_reader.converter.get_padding_right(tokens_as_string=True)

    elif args.task == 'sslm':
        nn = SentimentModel.create_new(feature_tables, args.window, args.hidden, args.alpha)
        padding_left = text_reader.converter.get_padding_left(tokens_as_string=True)
        padding_right = text_reader.converter.get_padding_right(tokens_as_string=True)
        
    else:
        # pos, srl_predicates or ner
        num_tags = len(text_reader.tag_dict)
        nn = Network.create_new(feature_tables, args.window, args.hidden, num_tags)

        padding_left = text_reader.converter.get_padding_left(args.task == 'pos' or args.task == 'ner')
        padding_right = text_reader.converter.get_padding_right(args.task == 'pos' or args.task == 'ner')
    
    nn.padding_left = np.array(padding_left)
    nn.padding_right = np.array(padding_right)
    nn.learning_rate = args.learning_rate
    nn.learning_rate_features = args.learning_rate_features
    nn.learning_rate_trans = args.learning_rate_transitions
    
    if args.task == 'lm':
        layer_sizes = (nn.input_size, nn.hidden_size, 1)
    elif args.task == 'sslm':
        layer_sizes = (nn.input_size, nn.hidden_size, 2)
    elif  args.task.startswith('srl') and  args.convolution > 0 and args.hidden > 0:
        layer_sizes = (nn.input_size, nn.hidden_size, nn.hidden2_size, nn.output_size)
    else:
        layer_sizes = (nn.input_size, nn.hidden_size, nn.output_size)
    
    logger.info("Created new network with the following layer sizes: %s"
                % ', '.join(str(x) for x in layer_sizes))
    
    return nn
        
def save_features(nn, md):
    """
    Receives a sequence of feature tables and saves each one in the appropriate file.
    
    :param nn: the neural network
    :param md: a Metadata object describing the network
    """
    iter_tables = iter(nn.feature_tables)
    # word features
    utils.save_features_to_file(iter_tables.next(), config.FILES[md.type_features])
    
    # other features - the order is important!
    if md.use_caps:
        utils.save_features_to_file(iter_tables.next(), config.FILES[md.caps_features])
    if md.use_suffix:
        utils.save_features_to_file(iter_tables.next(), config.FILES[md.suffix_features])
    if md.use_prefix:
        utils.save_features_to_file(iter_tables.next(), config.FILES[md.prefix_features])
    if md.use_pos:
        utils.save_features_to_file(iter_tables.next(), config.FILES[md.pos_features])
    if md.use_chunk:
        utils.save_features_to_file(iter_tables.next(), config.FILES[md.chunk_features])

    # NER gazetteer features
    if md.use_gazetteer:
        for file in config.FILES[md.gaz_features]:
            utils.save_features_to_file(iter_tables.next(), file)
    
def load_network_train(args, md):
    """Loads and returns a neural network with all the necessary data."""
    nn = taggers.load_network(md)
    
    logger.info("Loaded network with following parameters:")
    logger.info(nn.description())
    
    nn.learning_rate = args.learning_rate
    nn.learning_rate_features = args.learning_rate_features
    if md.task != 'lm' and md.task != 'sslm':
        nn.learning_rate_trans = args.learning_rate_transitions
    
    return nn

def train(text_reader, args):   # was reader. Attardi
    """Trains a neural network for the selected task."""
    report_intervals = max(args.iterations / 200, 1)
    np.seterr(over='raise')
    
    if args.task.startswith('srl') and args.task != 'srl_predicates':
        arg_limits = None if args.task != 'srl_classify' else text_reader.arg_limits
        
        nn.train(text_reader.sentences, text_reader.predicates, text_reader.tags, 
                 args.iterations, report_intervals, args.accuracy, arg_limits)
    elif args.task == 'lm':
        report_intervals = 10000
        nn.train(text_reader.sentences, args.iterations, report_intervals)
    elif args.task == 'sslm':
        report_intervals = 10000
        nn.train(text_reader.sentences, args.iterations, report_intervals, text_reader.polarities, text_reader.word_dict)
    else:
        nn.train(text_reader.sentences, text_reader.tags, 
                 args.iterations, report_intervals, args.accuracy)

def saver(nn_file, md):
    """Function to save model periodically"""
    def save(nn):
        save_features(nn, md)
        nn.save(nn_file)
    return save

if __name__ == '__main__':
    args = arguments.get_args()

    # set the seed for replicability
    #np.random.seed(42)

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    utils.set_logger(logging_level)
    logger = logging.getLogger("Logger")

    config.set_data_dir(args.data)

    use_caps = getattr(args, 'caps', False)
    use_suffix = getattr(args, 'suffix', False)
    use_prefix = getattr(args, 'prefix', False)
    use_pos = getattr(args, 'pos', False)
    use_chunk = getattr(args, 'chunk', False)
    use_lemma = getattr(args, 'lemma', False)
    use_gazetteer = getattr(args, 'gazetteer', False)
    
    if not args.load_network:
        # if we are about to create a new network, create the metadata too
        md = metadata.Metadata(args.task, use_caps, use_suffix, use_prefix, use_pos, use_chunk, use_lemma, use_gazetteer)
        md.save_to_file()
    else:
        md = metadata.Metadata.load_from_file(args.task)
    
    text_reader = create_reader(args, md)
    
    text_reader.create_converter()
    text_reader.codify_sentences()
    
    if args.load_network:
        logger.info("Loading provided network...")
        nn = load_network_train(args, md)
    else:
        logger.info('Creating new network...')
        feature_tables = utils.load_features(args, md, text_reader)
        nn = create_network(args, text_reader, feature_tables, md)
    
    logger.info("Starting training with %d sentences" % len(text_reader.sentences))
    logger.info("Network weights learning rate: %f" % nn.learning_rate)
    logger.info("Feature vectors learning rate: %f" % nn.learning_rate_features)
    logger.info("Tag transition matrix learning rate: %f" % nn.learning_rate_trans)
    
    filename = config.FILES[md.network]
    nn.saver = saver(filename, md)

    train(text_reader, args)
    
    logger.info("Saving trained models...")
    save_features(nn, md)
    
    nn.save(filename)
    logger.info("Saved network to %s" % nn_file)
    
