# -*- coding: utf-8 -*-

"""
Script for argument parsing and a few verifications. 
These arguments used by the training script.
"""

import argparse

def fill_defaults(args, defaults_per_task):
    """
    This function fills arguments not explicitly set (left as None)
    with default values according to the chosen task.
    
    We can't rely on argparse to it because using subparsers with
    set_defaults and a parent parser overwrites the defaults. 
    """
    task = args.task
    if task not in defaults_per_task:
        return
    defaults = defaults_per_task[task]
    for arg in args.__dict__:
        if getattr(args, arg) is None and arg in defaults:
            setattr(args, arg, defaults[arg])

def get_args():
    parser = argparse.ArgumentParser(description="Trains a neural\
        network for a specified task.")
    subparsers = parser.add_subparsers(title='Tasks',
                                       dest='task',
                                       description='Task to train nlpnet for. '\
                                       'Type %(prog)s [TASK] -h to get task-specific help.')
    
    defaults = {}
    
    # parser with arguments shared among all tasks
    # each task-specific parser may define defaults
    base_parser = argparse.ArgumentParser(add_help=False)
    
    base_parser.add_argument('-w', '--window', type=int, default=5,
                             help='Size of the word window (default 5)',
                             dest='window')
    base_parser.add_argument('-f', '--num_features', type=int, default=50,
                             help='Number of features per word (default 50)',
                             dest='num_features')
    base_parser.add_argument('--load_features', action='store_true',
                             help="Load previously saved word type features "\
                             "(overrides -f and must also \
                             load a dictionary file)", dest='load_types')
    base_parser.add_argument('--load_network', action='store_true',
                             help='Load previously saved network')
    base_parser.add_argument('-e', '--epochs', type=int, default=100,
                             help='Number of training epochs (default 100)',
                             dest='iterations')
    base_parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
                             help='Learning rate for network weights (default 0.001)',
                             dest='learning_rate')
    base_parser.add_argument('--lf', type=float, default=0.01,
                             help='Learning rate for features (default 0.01)',
                             dest='learning_rate_features')
    base_parser.add_argument('--lt', type=float, default=0.01,
                             help='Learning rate for transitions (default 0.01)',
                             dest='learning_rate_transitions')
    base_parser.add_argument('-a', '--accuracy', type=float, default=0.0,
                             help='Desired accuracy per tag.',
                             dest='accuracy')
    base_parser.add_argument('-n', '--hidden', type=int, default=200,
                             help='Number of hidden neurons (default 200)',
                             dest='hidden')
    base_parser.add_argument('-v', '--verbose', help='Verbose mode',
                             action="store_true")
    base_parser.add_argument('--gold', type=str, default=None,
                             help='File with annotated data for training.', required=True)
    base_parser.add_argument('--data', type=str, default=None,
                             help='Directory to save new models and load partially trained ones', required=True)
    base_parser.add_argument('--variant', type=str, default=None,
                             help='If "polyglot" use Polyglot case conventions; if "senna" use SENNA conventions.')

    # parser with arguments shared among convolutional-based tasks
    conv_parser = argparse.ArgumentParser(add_help=False)
    conv_parser.add_argument('-c', '--convolution', type=int,
                             help='Number of convolution neurons',
                             dest='convolution')
    conv_parser.add_argument('--pos', const=5, nargs='?', type=int, default=None,
                             help='Include part-of-speech features. Optionally, supply the number of features (default 5)')
    conv_parser.add_argument('--max_dist', type=int, default=10,
                             help='Maximum distance to have a separate feature')
    conv_parser.add_argument('--target_features', type=int, default=5,
                             help='Number of features for distance to target word')
    conv_parser.add_argument('--pred_features', type=int, default=5,
                             help='Number of features for distance to predicate')

    # POS argument parser
    pos_parser = subparsers.add_parser('pos', help='POS tagging', 
                                       parents=[base_parser])
    pos_parser.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                             help='Include capitalization features. Optionally, supply the number of features (default 5)')
    pos_parser.add_argument('--suffix', const=5, nargs='?', type=int, default=None,
                            help='Include suffix features. Optionally, supply the number of features (default 5)')
    pos_parser.add_argument('--suffix_size', type=int, default=5,
                            help='Use suffixes up to this size (in characters, default 5). '\
                            'Only used if --suffix is supplied')
    pos_parser.add_argument('--prefix', const=2, nargs='?', type=int, default=None,
                            help='Include prefix features. Optionally, '\
                            'supply the number of features (default 2)')
    pos_parser.add_argument('--prefix_size', type=int, default=5,
                            help='Use prefixes up to this size (in characters, default 5). '\
                            'Only used if --suffix is supplied')
    defaults['pos'] = dict(window=5, hidden=100, iterations=15, 
                           learning_rate=0.001, learning_rate_features=0.001,
                           learning_rate_transitions=0.001)
    
    # SRL argument parser
    # There is another level of subparsers for predicate detection / 
    # argument boundary identification / argument classification / 
    # (id + class) in one step
    srl_parser = subparsers.add_parser('srl', help='Semantic Role Labeling',
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    srl_parser.set_defaults(identify=False, predicates=False, classify=False)

    desc = '''SRL has 3 steps: predicate  detection, argument identification and 
argument classification. Each one depends on the one before.

You need one model trained for each subtask (or one for predicate
detection and another with the other 2 steps) in order to perform
full SRL.

Type %(prog)s [SUBTASK] -h to get subtask-specific help.'''
    
    srl_subparsers = srl_parser.add_subparsers(title='SRL subtasks',
                                               dest='subtask',
                                               description=desc)
    srl_subparsers.add_parser('pred', help='Predicate identification',
                              parents=[base_parser])
    defaults['srl_predicates'] = dict(window=5, hidden=50, iterations=1, 
                                      learning_rate=0.01, learning_rate_features=0.01,
                                      learning_rate_transitions=0.01,
                                      predicates=True)
    
    srl_subparsers.add_parser('id', help='Argument identification',
                              parents=[base_parser, conv_parser])
    defaults['srl_boundary'] = dict(window=3, hidden=150, convolution=150, 
                                    identify=True, iterations=15,
                                    learning_rate=0.001, learning_rate_features=0.001,
                                    learning_rate_transitions=0.001)
    
    srl_subparsers.add_parser('class', help='Argument classification',
                              parents=[base_parser, conv_parser])
    defaults['srl_classify'] = dict(window=3, hidden=0, convolution=100, 
                                    classify=True, iterations=3,
                                    learning_rate=0.01, learning_rate_features=0.01,
                                    learning_rate_transitions=0.01)
    srl_subparsers.add_parser('1step', parents=[base_parser, conv_parser],
                              help='Argument identification and '\
                              'classification together')
    defaults['srl'] = dict(window=3, hidden=150, convolution=200, iterations=15,
                           learning_rate=0.001, learning_rate_features=0.001,
                           learning_rate_transitions=0.001)
    
    # NER argument parser
    ner_parser = subparsers.add_parser('ner', help='NER tagging', 
                                       parents=[base_parser])
    ner_parser.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                             help='Include capitalization features. Optionally, supply the number of features (default 5)')
    ner_parser.add_argument('--use_lemma', action='store_true', dest='use_lemma',
                            help='Use word lemmas instead of surface forms.')
    ner_parser.add_argument('--gazetteer', const=5, nargs='?', type=int, default=None,
                            help='Include gazetteer features. Optionally, supply the number of features (default 5)')

    # LM argument parser
    lm_parser = subparsers.add_parser('lm', help='Language Model creation', 
                                      parents=[base_parser])
    lm_parser.add_argument('--dict_size',type=int, default=100000,
                           help='Size of embeddings dictionary (default 100000)')

    # SSLM argument parser
    sslm_parser = subparsers.add_parser('sslm', help='Sentiment Specific Language Model creation', 
                                      parents=[base_parser])
    sslm_parser.add_argument('--dict_size',type=int, default=100000,
                           help='Size of embeddings dictionary (default 100000)')
    sslm_parser.add_argument('--ngrams',type=int, default=1,
                           help='Length of ngrams to consider (default 1)')
    sslm_parser.add_argument('--alpha', type=float, default=0.5,
                           help='Weight of syntactic loss (default 0.5)',
                           dest='alpha')

    args = parser.parse_args()
    if args.task == 'srl':
        if args.subtask == 'class':
            args.task = 'srl_classify'
            args.classify = True
        elif args.subtask == 'id':
            args.task = 'srl_boundary'
            args.identify = True
        elif args.subtask == 'pred':
            args.task = 'srl_predicates'
            args.predicates = True
    
    fill_defaults(args, defaults)
    return args
