# -*- coding: utf-8 -*-

"""
This script contains the definition of the Metadata class.
It can also be invoked in order to create a Metadata object
and save it to a file in the data directory.
"""

import cPickle

import config

class Metadata(object):
    """
    Class for storing metadata about a neural network and its 
    parameter files.
    """
    
    def __init__(self, task, use_caps=True, use_suffix=False,
                 use_prefix=False, use_pos=False,
                 use_chunk=False, use_lemma=False, use_gazetteer=False):
        self.task = task
        self.paths = config.FILES
        self.use_caps = use_caps
        self.use_suffix = use_suffix
        self.use_prefix = use_prefix
        self.use_pos = use_pos
        self.use_chunk = use_chunk
        self.use_lemma = use_lemma
        self.use_gazetteer = use_gazetteer
        self.metadata = '%s_metadata' % task
        self.network = '%s_network' % task
        
        if task != 'lm' and task != 'sslm':
            self.tag_dict = '%s_tag_dict' % task
        else:
            self.tag_dict = None
        
        if task == 'srl_boundary':
            self.pred_dist_table = 'srl_boundary_pred_dist_table'
            self.target_dist_table = 'srl_boundary_target_dist_table'
            self.transitions = 'srl_boundary_srl_transitions'
            self.type_features = 'srl_boundary_type_features'
            self.caps_features = 'srl_boundary_caps_features'
            self.pos_features = 'srl_boundary_pos_features'
            self.chunk_features = 'srl_boundary_chunk_features'
            self.suffix_features = None
            
        elif task == 'srl_classify':
            self.pred_dist_table = 'pred_dist_table'
            self.target_dist_table = 'srl_classify_target_dist_table'
            self.transitions = None
            self.type_features = 'srl_classify_type_features'
            self.caps_features = 'srl_classify_caps_features'
            self.pos_features = 'srl_classify_pos_features'
            self.chunk_features = 'srl_classify_chunk_features'
            self.suffix_features = None
        
        elif task == 'srl':
            # one step srl
            self.pred_dist_table = 'srl_1step_pred_dist_table'
            self.target_dist_table = 'srl_1step_target_dist_table'
            self.transitions = 'srl_1step_srl_transitions'
            self.type_features = 'srl_1step_type_features'
            self.caps_features = 'srl_1step_caps_features'
            self.pos_features = 'srl_1step_pos_features'
            self.chunk_features = 'srl_1step_chunk_features'
            self.suffix_features = None
        
        else:
            self.type_features = '%s_type_features' % task
            self.caps_features = '%s_caps_features' % task
            self.pos_features = '%s_pos_features' % task
            self.chunk_features = '%s_chunk_features' % task
            self.suffix_features = '%s_suffix_features' % task

        if task == 'ner':
            self.gazetteer = 'ner_gazetteer' # gazetteer file
            self.gaz_features = 'ner_gaz_features'
            self.gaz_classes = ['LOC', 'MISC', 'ORG', 'PER']
    
    def __str__(self):
        """Shows the task at hand and which attributes are used."""
        lines = []
        lines.append("Metadata for task %s" % self.task)
        for k in self.__dict__:
            if isinstance(k, str) and k.startswith('use_'):
                lines.append('%s: %s' % (k, self.__dict__[k]))
        
        return '\n'.join(lines)
    
    def save_to_file(self): 
        """
        Save the contents of the metadata to a file. The filename is determined according
        to the task.
        """
        filename = self.paths[self.metadata]
        with open(filename, 'wb') as f:
            cPickle.dump(self.__dict__, f, 2)
    
    @classmethod
    def load_from_file(cls, task, paths = None):
        """
        Reads the file containing the metadata for the given task and returns a 
        Metadata object.
        """
        if paths is None:
            paths = config.FILES
        md = Metadata(task, paths)

        # the actual content of the file is the __dict__ member variable,
        # which contain all the instance's data
        filename = paths[md.metadata]
        with open(filename, 'rb') as f:
            data = cPickle.load(f)
        md.__dict__.update(data)
        
        return md

