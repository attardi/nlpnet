# -*- coding: utf-8 -*- 

"""
Configuration data for the system.
"""

import os

data_dir = None
FILES = {}

def get_config_paths(directory):
    """Sets the data directory containing the data for the models."""
    assert os.path.isdir(directory), 'Invalid data directory'

    dict = { key: os.path.join(directory, value) for key, value in [ 
        # cross-task data
        ('.', '.'), #for data_dir access
        ('vocabulary'                  , 'vocabulary.txt'),
        ('word_dict_dat'               , 'vocabulary.txt'), # deprecated
        ('type_features'               , 'types-features.npy'),
        ('termvectors'                 , 'termvectors.txt'),

        # Language Model
        ('network_lm'			, 'lm-network.npz'),
        ('metadata_lm'			, 'lm-metadata.pickle'),
        ('type_features_lm'		, 'lm-embeddings.npy'),

        # Sentiment Model
        ('network_sslm'			, 'sslm-network.npz'),
        ('metadata_sslm'		, 'sslm-metadata.pickle'),
        ('type_features_sslm'		, 'sslm-embeddings.npy'),

        # POS
        ('network_pos'                 , 'pos-network.npz'),
        ('pos_tags'                    , 'pos-tags.txt'),
        ('pos_tag_dict'                , 'pos-tags.txt'),
        ('suffixes'                    , 'suffixes.txt'),
        ('prefixes'                    , 'prefixes.txt'),
        ('metadata_pos'                , 'pos-metadata.pickle'),
        ('type_features_pos'           , 'pos-types-features.npy'),
        ('caps_features_pos'           , 'pos-caps-features.npy'),
        ('suffix_features_pos'         , 'pos-suffix-features.npy'),
        ('prefix_features_pos'         , 'pos-prefix-features.npy'),

        # NER
        ('network_ner'		, 'ner-network.npz'),
        ('ner_tag_dict'		, 'ner-tag-dict.pickle'),
        ('type_features_ner'	, 'ner-types-features.npy'),
        ('caps_features_ner'	, 'ner-caps-features.npy'),
        ('suffix_features_ner'	, 'ner-suffix-features.npy'),
        ('gazetteer_ner'	, 'eng.list'),

        # chunk
        ('chunk_tag_dict'              , 'chunk-tag-dict.pickle'),
        ('chunk_tags'                  , 'chunk-tags.txt'),

        # SRL
        ('network_srl'                 , 'srl-network.npz'),
        ('network_srl_boundary'        , 'srl-id-network.npz'),
        ('network_srl_classify'        , 'srl-class-network.npz'),
        ('network_srl_predicates'      , 'srl-class-predicates.npz'),
        ('srl_iob_tag_dict'            , 'srl-tags.txt'),
        ('srl_iob_tags'                , 'srl-tags.txt'),
        ('srl_tags'                    , 'srl-tags.txt'),
        ('srl_classify_tag_dict'       , 'srl-tags.txt'),
        ('srl_classify_tags'           , 'srl-tags.txt'),
        ('srl_predicates_tag_dict'     , 'srl-predicates-tags.txt'),
        ('srl_predicates_tags'         , 'srl-predicates-tags.txt'),
        ('type_features_boundary'      , 'types-features-id.npy'),
        ('caps_features_boundary'      , 'caps-features-id.npy'),
        ('pos_features_boundary'       , 'pos-features-id.npy'),
        ('chunk_features_boundary'     , 'chunk-features-id.npy'),
        ('type_features_classify'      , 'types-features-class.npy'),
        ('caps_features_classify'      , 'caps-features-class.npy'),
        ('pos_features_classify'       , 'pos-features-class.npy'),
        ('chunk_features_classify'     , 'chunk-features-class.npy'),
        ('type_features_1step'         , 'types-features-1step.npy'),
        ('caps_features_1step'         , 'caps-features-1step.npy'),
        ('pos_features_1step'          , 'pos-features-1step.npy'),
        ('chunk_features_1step'        , 'chunk-features-1step.npy'),
        ('type_features_srl_predicates', 'types-features-preds.npy'),
        ('caps_features_srl_predicates', 'caps-features-preds.npy'),
        ('pos_features_srl_predicates' , 'pos-features-preds.npy'),
        ('metadata_srl'                , 'srl-metadata.pickle'),
        ('metadata_srl_boundary'       , 'srl-metadata-boundary.pickle'),
        ('metadata_srl_classify'       , 'srl-metadata-classify.pickle'),
        ('metadata_srl_predicates'     , 'srl-metadata-predicates.pickle'),
        ]
    }
    # NER
    dict['gaz_features_ner'] = [os.path.join(data_dir, 'ner-gazl-features.npy'),
                              os.path.join(data_dir, 'ner-gazm-features.npy'),
                              os.path.join(data_dir, 'ner-gazo-features.npy'),
                              os.path.join(data_dir, 'ner-gazp-features.npy')]
    return dict

def set_data_dir(directory):
    """Sets the global data directory containing the data for the models."""
    global data_dir, FILES
    data_dir = directory
    FILES = get_config_paths(directory)
