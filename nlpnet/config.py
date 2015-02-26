# -*- coding: utf-8 -*- 

"""
Configuration data for the system.
"""

import os

data_dir = None
FILES = {}

def get_config_paths(directory):
    """Sets the data directory containing the data for the models."""
    assert os.path.isdir(directory), 'Invalid data directory: ' + directory

    dict = { key: os.path.join(directory, value) for key, value in [ 
        # cross-task data
        ('.', '.'), #for data_dir access
        ('vocabulary'                  , 'vocabulary.txt'),
        ('type_features'               , 'vectors.npy'),

        # Language Model
        ('lm_network'			, 'lm-network.npz'),
        ('lm_metadata'			, 'lm-metadata.pickle'),
        ('lm_type_features'		, 'lm-vectors.npy'),

        # Sentiment Model
        ('sslm_network'			, 'sslm-network.npz'),
        ('sslm_metadata'		, 'sslm-metadata.pickle'),
        ('sslm_type_features'		, 'sslm-vectors.npy'),

        # POS
        ('pos_network'                 , 'pos-network.npz'),
        ('pos_tag_dict'                , 'pos-tags.txt'),
        ('suffixes'                    , 'suffixes.txt'),
        ('prefixes'                    , 'prefixes.txt'),
        ('pos_metadata'                , 'pos-metadata.pickle'),
        ('pos_type_features'           , 'pos-word-vectors.npy'),
        ('pos_caps_features'           , 'pos-caps-vectors.npy'),
        ('pos_suffix_features'         , 'pos-suffix-vectors.npy'),
        ('pos_prefix_features'         , 'pos-prefix-vectors.npy'),

        # NER
        ('ner_metadata'		, 'ner-metadata.pickle'),
        ('ner_network'		, 'ner-network.npz'),
        ('ner_tag_dict'		, 'ner-tag-dict.txt'),
        ('ner_type_features'	, 'ner-word-vectors.npy'),
        ('ner_pos_features'	, 'ner-pos-vectors.npy'),
        ('ner_caps_features'	, 'ner-caps-vectors.npy'),
        ('ner_suffix_features'	, 'ner-suffix-vectors.npy'),
        ('ner_gazetteer'	, 'eng.list'),

        # chunk
        ('chunk_tag_dict'              , 'chunk-tag-dict.txt'),
        ('chunk_tags'                  , 'chunk-tags.txt'),

        # SRL
        ('srl_network'                 , 'srl-network.npz'),
        ('srl_network_boundary'        , 'srl-id-network.npz'),
        ('srl_network_classify'        , 'srl-class-network.npz'),
        ('srl_network_predicates'      , 'srl-class-predicates.npz'),
        ('srl_iob_tag_dict'            , 'srl-tags.txt'),
        ('srl_iob_tags'                , 'srl-tags.txt'),
        ('srl_tags'                    , 'srl-tags.txt'),
        ('srl_classify_tag_dict'       , 'srl-tags.txt'),
        ('srl_classify_tags'           , 'srl-tags.txt'),
        ('srl_predicates_tag_dict'     , 'srl-predicates-tags.txt'),
        ('srl_predicates_tags'         , 'srl-predicates-tags.txt'),
        ('srl_boundary_type_features'      , 'types-vectors-id.npy'),
        ('srl_boundary_caps_features'      , 'caps-vectors-id.npy'),
        ('_boundary_pos_features'       , 'pos-vectors-id.npy'),
        ('srl_boundary_chunk_features'     , 'chunk-vectors-id.npy'),
        ('srl_classify_type_features'      , 'types-vectors-class.npy'),
        ('srl_classify_caps_features'      , 'caps-vectors-class.npy'),
        ('srl_classify_pos_features'       , 'pos-vectors-class.npy'),
        ('srl_classify_chunk_features'     , 'chunk-vectors-class.npy'),
        ('srl_1step_type_features'         , 'types-vectors-1step.npy'),
        ('srl_1step_caps_features'         , 'caps-vectors-1step.npy'),
        ('srl_1step_pos_features'          , 'pos-vectors-1step.npy'),
        ('srl_1step_chunk_features'        , 'chunk-vectors-1step.npy'),
        ('srl_predicates_type_features', 'types-vectors-preds.npy'),
        ('srl_predicates_caps_features', 'caps-vectors-preds.npy'),
        ('srl_predicates_pos_features' , 'pos-vectors-preds.npy'),
        ('srl_metadata'                , 'srl-metadata.pickle'),
        ('srl_boundary_metadata'       , 'srl-metadata-boundary.pickle'),
        ('srl_metadata_classify'       , 'srl-metadata-classify.pickle'),
        ('srl_predicates_metadata'     , 'srl-metadata-predicates.pickle'),
        ]
    }
    # NER
    dict['ner_gaz_features'] = [os.path.join(data_dir, 'ner-gazl-vectors.npy'),
                              os.path.join(data_dir, 'ner-gazm-vectors.npy'),
                              os.path.join(data_dir, 'ner-gazo-vectors.npy'),
                              os.path.join(data_dir, 'ner-gazp-vectors.npy')]
    return dict

def set_data_dir(directory):
    """Sets the global data directory containing the data for the models."""
    global data_dir, FILES
    data_dir = directory
    FILES = get_config_paths(directory)
