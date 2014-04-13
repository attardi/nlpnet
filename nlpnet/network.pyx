# -*- coding: utf-8 -*-
#cython: embedsignature=True

"""
A neural network for NLP tagging tasks.
It employs feature tables to store feature vectors for each token.
"""

import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool
import math
from itertools import izip
import logging

ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t
ctypedef Py_ssize_t INDEX_t

cdef hardtanh(np.ndarray weights):
    cdef int i
    cdef float w
    for i in range(len(weights)):
        w = weights[i]
        if w < -1:
            weights[i] = -1
        elif w > 1:
            weights[i] = 1

cdef hardtanhd(np.ndarray[FLOAT_t, ndim=2] weights):
    """derivative of hardtanh"""
    cdef np.ndarray out = np.zeros_like(weights)
    cdef int i
    cdef float w
    for i, w in enumerate(weights.flat):
        if -1.0 < w < 1.0:
            out.flat[i] = 1.0
    return out

cdef class Network:
    
    # sizes and learning rates
    cdef readonly int word_window_size, input_size, hidden_size, output_size
    cdef public float learning_rate, learning_rate_features
    
    # padding stuff
    cdef np.ndarray padding_left, padding_right
    cdef public np.ndarray pre_padding, pos_padding
    
    # weights, biases, calculated values
    cdef readonly np.ndarray hidden_weights, output_weights
    cdef readonly np.ndarray hidden_bias, output_bias
    cdef readonly np.ndarray input_values, hidden_values
    
    # feature tables 
    cdef public list feature_tables
    
    # transitions
    cdef public float learning_rate_trans
    cdef public np.ndarray transitions
    
    # the score for a given path
    cdef readonly float answer_score
    
    # gradients
    cdef readonly np.ndarray net_gradients, trans_gradients
    cdef readonly np.ndarray input_sent_values, hidden_sent_values
    
    # data for statistics during training. 
    cdef float error, accuracy, float_errors
    cdef int train_items, skips
    
    # function to save periodically
    cdef public object saver

    @classmethod
    def create_new(cls, feature_tables, int word_window, int hidden_size, 
                 int output_size):
        """
        Creates a new neural network.
        """
        # sum the number of features in all tables 
        cdef int input_size = sum(table.shape[1] for table in feature_tables)
        input_size *= word_window
        
        # creates the weight matrices
        # all weights are between -0.1 and +0.1
        # hidden_weights = 0.2 * np.random.random((hidden_size, input_size)) - 0.1
        # hidden_bias = 0.2 * np.random.random(hidden_size) - 0.1
        # output_weights = 0.2 * np.random.random((output_size, hidden_size)) - 0.1
        # output_bias = 0.2 * np.random.random(output_size) - 0.1

	# centered uniform distribution with variance = 1/sqrt(fanin)
	# variance = 1/12 interval ^ 2
	# interval = 3.46 / fanin ^ 1/4
        #high = 1.732 / np.power(input_size, 0.25) # 0.416
        #high = 2.38 / np.sqrt(input_size) # [Bottou-88]
        high = 0.1
        hidden_weights = np.random.uniform(-high, high, (hidden_size, input_size))
        hidden_bias = np.random.uniform(-high, high, (hidden_size))
        #high = 1.732 * np.power(hidden_size, 0.25)
        #high = 2.38 * np.sqrt(hidden_size) # [Bottou-88]
        high = 0.1
        output_weights = np.random.uniform(-high, high, (output_size, hidden_size))
        output_bias = np.random.uniform(-high, high, (output_size))
        
        net = Network(word_window, input_size, hidden_size, output_size,
                      hidden_weights, hidden_bias, output_weights, output_bias)
        net.feature_tables = feature_tables
        
        return net
        
    def __init__(self, word_window, input_size, hidden_size, output_size,
                 hidden_weights, hidden_bias, output_weights, output_bias):
        """
        This function isn't expected to be directly called.
        Instead, use the classmethods load_from_file or 
        create_new.
        """
        self.learning_rate = 0
        self.learning_rate_features = 0
        
        self.word_window_size = word_window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # +1 is due for initial transition
        # A_i_j score for jumping from tag i to j
        # A_0_i = transitions[-1]
        self.transitions = np.zeros((self.output_size + 1, self.output_size))
#        self.transitions = None # trigger WLL

        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias

	# Attardi: saver fuction
        self.saver = lambda nn: None
    
    def description(self):
        """
        Returns a textual description of the network.
        """
        table_dims = [str(t.shape[1]) for t in self.feature_tables]
        table_dims =  ', '.join(table_dims)
        
        desc = """
Word window size: %d
Feature table sizes: %s
Input layer size: %d
Hidden layer size: %d
Output size: %d
""" % (self.word_window_size, table_dims, self.input_size, self.hidden_size, self.output_size)
        
        return desc
    
    
    def run(self, np.ndarray indices):
        """
        Runs the network for a given input. 
        
        :param indices: a 2-dim np array of indices to the feature tables.
            Each element must have the indices to each feature table.
        """
        # find the actual input values concatenating the feature vectors
        # for each input token
        cdef np.ndarray input_data
        input_data = np.concatenate(
                                    [table[index] 
                                     for token_indices in indices
                                     for index, table in izip(token_indices, 
                                                              self.feature_tables)
                                     ]
                                    )

        # store the output in self in order to use in the backprop
        self.input_values = input_data
        # (hidden_size, input_size) . input_size = hidden_size
        self.hidden_values = self.hidden_weights.dot(input_data)
        self.hidden_values += self.hidden_bias
        #self.hidden_values = np.tanh(self.hidden_values)
        # senna. Attardi
        hardtanh(self.hidden_values)
        
        cdef np.ndarray output = self.output_weights.dot(self.hidden_values)
        output += self.output_bias
        
        return output
    
    property padding_left:
        """
        The padding element filling the "void" before the beginning
        of the sentence.
        """
        def __get__(self):
            return self.padding_left
    
        def __set__(self, np.ndarray padding_left):
            self.padding_left = padding_left
            self.pre_padding = np.array((self.word_window_size / 2) * [padding_left])
    
    property padding_right:
        """
        The padding element filling the "void" after the end
        of the sentence.
        """
        def __get__(self):
            return self.padding_right
    
        def __set__(self, np.ndarray padding_right):
            self.padding_right = padding_right
            self.pos_padding = np.array((self.word_window_size / 2) * [padding_right])
    
    def tag_sentence(self, np.ndarray sentence):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        
        :param sentence: a 2-dim numpy array, where each item encodes a token.
        """
        scores = self._tag_sentence(sentence, train=False)
        # computes full score, combining ftheta and A (if SLL)
        return self._viterbi(scores)

    def _tag_sentence(self, np.ndarray sentence, bool train=False, tags=None):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        
        :param sentence: a 2-dim numpy array, where each item encodes a token.
        :param train: if True, perform weight and feature correction.
        :param tags: the correct tags (needed when training)
        :return: the scores for all tokens
        """
        cdef np.ndarray answer
        # scores[t, i] = ftheta_i,t = score for i-th tag, t-th word
        cdef np.ndarray scores = np.empty((len(sentence), self.output_size))
        
        if train:
            self.input_sent_values = np.empty((len(sentence), self.input_size))
            self.hidden_sent_values = np.empty((len(sentence), self.hidden_size))
        
        # add padding to the sentence
        cdef np.ndarray padded_sentence = np.vstack((self.pre_padding,
                                                     sentence,
                                                     self.pos_padding))

        # run through all windows in the sentence
        for i in xrange(len(sentence)):
            window = padded_sentence[i: i+self.word_window_size]
            scores[i] = self.run(window)
            if train:
                self.input_sent_values[i] = self.input_values
                self.hidden_sent_values[i] = self.hidden_values 
        
        if train:
            if self._calculate_gradients_all_tokens(tags, scores):
#            if self._calculate_gradients_wll(tags, scores):
                self._backpropagate(sentence)
         
        return scores
    
    def _calculate_all_scores(self, scores):
        """
        Calculates a matrix with the scores for all possible paths at all given
        points (tokens).
        In the returned matrix, delta[i][j] means the sum of all scores 
        ending in token i with tag j (delta_i(j) in eq. 14 in the paper)
        """
        # logadd for first token. the transition score of the starting tag must be used.
        # it turns out that logadd = log(exp(score)) = score
        # (use long double because taking exp's leads to very very big numbers)
        # scores[t][k] = ftheta_k,t
        delta = np.longdouble(scores)
        # transitions[len(sentence)] represents initial transition, A_0,i in paper (mispelled as A_i,0)
        # delta_0(k) = ftheta_k,0 + A_0,i
        delta[0] += self.transitions[-1]
        
        # logadd for the remaining tokens
        # delta_t(k) = ftheta_k,t + logadd_i(delta_t-1(i) + A_i,k)
        #            = ftheta_k,t + log(Sum_i(exp(delta_t-1(i) + A_i,k)))
        transitions = self.transitions[:-1].T # A_k,i
        for token in xrange(1, len(delta)):
            # sum by rows
            logadd = np.log(np.sum(np.exp(delta[token - 1] + transitions), 1))
            delta[token] += logadd
            
        return delta
    
    def _calculate_gradients_wll(self, tags, scores):
        """
        The aim is to minimize the word-level log-likelihood:
        C(ftheta) = logadd_j(ftheta_j) - ftheta_y,
        where y is the sequence of correct tags
        
        :returns: if True, normal gradient calculation was performed.
            If False, the error was too low and weight correction should be
            skipped.
        """
        # initialize gradients
        # ((len(sentence), self.output_size)) // Attardi
        # dC / dftheta.T
        self.net_gradients = np.zeros_like(scores, np.float)

        # compute the negative gradient with respect to ftheta
        # dC / dftheta_i) = e(ftheta_i)/Sum_k(e(ftheta_k))
        exponentials = np.exp(scores)
        self.net_gradients = -(exponentials.T / exponentials.sum(1)).T

        # correct path and its gradient
        correct_path_score = 0
        token = 0
        for tag, net_scores in izip(tags, scores):
            self.net_gradients[token][tag] += 1 # negative gradient
            token += 1
            correct_path_score += net_scores[tag]

        # C(ftheta) = logadd_j(ftheta_j) - score(correct path)
        error = np.log(np.sum(np.exp(scores))) - correct_path_score
        self.error += error

        return True

    @cython.boundscheck(False)
    def _calculate_gradients_all_tokens(self, tags, scores):
        """
        Calculates the output and transition deltas for each token.
        The aim is to minimize the cost:
        C(ftheta,A) = logadd(scores for all possible paths) - score(correct path)
        
        :returns: if True, normal gradient calculation was performed.
            If False, the error was too low and weight correction should be
            skipped.
        """
        cdef np.ndarray delta 
        
        # ftheta_i,t = score for i-th tag, t-th word
        # s = Sum_i(A_tags[i-1],tags[i] + ftheta_i,i), i = 0, len(sentence)
        correct_path_score = 0
        last_tag = self.output_size
        for tag, net_scores in izip(tags, scores):
            trans = 0 if self.transitions is None else self.transitions[last_tag, tag]
            correct_path_score += trans + net_scores[tag]
            last_tag = tag 
        
        # delta[t] = delta_t in equation (14)
        delta = self._calculate_all_scores(scores)
        # logadd_i(delta_T(i)) = log(Sum_i(exp(delta_T(i))))
        # Sentence-level Log-Likelihood (SLL)
        # C(ftheta,A) = logadd_j(s(x, j, theta, A)) - score(correct path)
        error = np.log(np.sum(np.exp(delta[-1]))) - correct_path_score
        self.error += error
        
        # if the error is too low, don't bother training (saves time and avoids
        # overfitting). An error of 0.01 means a log-prob of -0.01 for the right
        # tag, i.e., more than 99% probability
        # error 0.69 -> 50% probability for right tag (minimal threshold)
        # error 0.22 -> 80%
        # error 0.1  -> 90%
        if error <= 0.01:
            self.skips += 1
            return False
        
        # initialize gradients
        # ((len(sentence), self.output_size)) // Attardi
        # dC / dftheta.T
        self.net_gradients = np.zeros_like(scores, np.float)
        # dC / dA
        self.trans_gradients = np.zeros_like(self.transitions, np.float)
        
        # things get nasty from here
        # refer to the papers to understand what exactly is going on
        
        # compute the gradients for the last token
        # dC_logadd / ddelta_T(i) = e(delta_T(i))/Sum_k(e(delta_T(k)))
        exponentials = np.exp(delta[-1])
        exp_sum = np.sum(exponentials)
        # negative gradients
        self.net_gradients[-1] = -exponentials / exp_sum
        
        transitions_t = 0 if self.transitions is None else self.transitions[:-1].T
        
        # now compute the gradients for the other tokens, from last to first
        for token in range(len(scores) - 2, -1, -1):
            
            # matrix with the exponentials which will be used to find the gradients
            # sum the scores for all paths ending with each tag in token "token"
            # with the transitions from this tag to the next
            # e(delta_t-1(i)+A_i,j)
            # Obtained by transposing twice
            # [e(delta_t-1(i)+A_j,i)]T
            # (output_size, output_size)
            exp_matrix = np.exp(delta[token] + transitions_t).T
            
            # the sums of exps, used to calculate the softmax
            # sum the exponentials by column
            denominators = np.sum(exp_matrix, 0)
            
            # softmax is the division of an exponential by the sum of all exponentials
            # (yields a probability)
            softmax = exp_matrix / denominators
            
            # multiply each value in the softmax by the gradient at the next tag
            # dC_logadd / ddelta_t(i) * softmax
            # Attardi: negative since net_gradients[token + 1] already negative
            grad_times_softmax = self.net_gradients[token + 1] * softmax
            self.trans_gradients[:-1, :] += grad_times_softmax
            
            # sum all transition gradients by row to find the network gradients
            # Sum_j(dC_logadd / ddelta_t(j) * softmax)
            # Attardi: negative since grad_times_softmax already negative
            self.net_gradients[token] = np.sum(grad_times_softmax, 1)
        
        # find the gradients for the starting transition
        # there is only one possibility to come from, which is the sentence start
        self.trans_gradients[-1] = self.net_gradients[0]
        
        # now, add +1 to the correct path
        last_tag = self.output_size
        for token, tag in enumerate(tags):
            self.net_gradients[token][tag] += 1 # negative gradient
            if self.transitions is not None:
                self.trans_gradients[last_tag][tag] += 1
            last_tag = tag
        
        return True
    
    @cython.boundscheck(False)
    def _viterbi(self, np.ndarray[FLOAT_t, ndim=2] scores):
        """
        Performs a Viterbi search over the scores for each tag using
        the transitions matrix. If a matrix wasn't supplied, 
        it will return the tags with the highest scores individually.
        """
        # pretty straightforward
        if self.transitions is None or len(scores) == 1:
            return scores.argmax(1)
            
        path_scores = np.empty_like(scores)
        path_backtrack = np.empty_like(scores, np.int)
        
        # now the actual Viterbi algorithm
        # first, get the scores for each tag at token 0
        # the last row of the transitions table has the scores for the first tag
        path_scores[0] = scores[0] + self.transitions[-1]
        
        output_range = np.arange(self.output_size) # outside loop. Attardi
        #for i, token in enumerate(scores[1:], 1):
        for i in xrange(1, len(scores)):
            
            # each line contains the score until each tag t plus the transition to each other tag t'
            prev_score_and_trans = (path_scores[i - 1] + self.transitions[:-1].T).T
            
            # find the previous tag that yielded the max score
            path_backtrack[i] = prev_score_and_trans.argmax(0)
            path_scores[i] = prev_score_and_trans[path_backtrack[i], 
                                                  output_range] + scores[i]
            
        # now find the maximum score for the last token and follow the backtrack
        answer = np.empty(len(scores), dtype=np.int)
        answer[-1] = path_scores[-1].argmax()
        self.answer_score = path_scores[-1][answer[-1]]
        previous_tag = path_backtrack[-1][answer[-1]]
        
        for i in range(len(scores) - 2, 0, -1):
            answer[i] = previous_tag
            previous_tag = path_backtrack[i][previous_tag]
        
        answer[0] = previous_tag
        return answer
    
    def train(self, list sentences, list tags, 
              int epochs, int epochs_between_reports=0,
              float desired_accuracy=0):
        """
        Trains the network to tag sentences.
        
        :param sentences: a list of 2-dim numpy arrays, where each item 
            encodes a sentence. Each item in a sentence has the 
            indices to its features.
        :param tags: a list of 1-dim numpy arrays, where each item has
            the tags of the sentences.
        :param epochs: number of training epochs
        :param epochs_between_reports: number of epochs to wait between
            reports about the training performance. 0 means no reports.
        :param desired_accuracy: training stops if the desired accuracy
            is reached. Ignored if 0.
        """
        logger = logging.getLogger("Logger")
        logger.info("Training for up to %d epochs" % epochs)
        top_accuracy = 0
        last_accuracy = 0
        min_error = np.Infinity 
        last_error = np.Infinity 
        
        np.seterr(all='raise')

        for i in xrange(epochs):
            self._train_epoch(sentences, tags)
            
            # Attardi: save model
            if self.error < min_error:
                min_error = self.error
                self.saver(self)

            if self.accuracy > top_accuracy:
                top_accuracy = self.accuracy
            
            if (epochs_between_reports > 0 and i % epochs_between_reports == 0) \
                or self.accuracy >= desired_accuracy > 0 \
                or (self.accuracy < last_accuracy and self.error > last_error):
                
                self._print_epoch_report(i + 1)
                
                if self.accuracy >= desired_accuracy > 0:
                    break
                
            last_accuracy = self.accuracy
            last_error = self.error
            
    def _print_epoch_report(self, int num):
        """
        Reports the status of the network in the given training
        epoch, including error and accuracy.
        """
        cdef float error = self.error / self.train_items
        logger = logging.getLogger("Logger")
        logger.info("%d epochs   Error: %f   Accuracy: %f   " \
            "%d corrections could be skipped   " \
            "%d floating point errors" % (num,
                                          error,
                                          self.accuracy,
                                          self.skips,
                                          self.float_errors))
    
    def _train_epoch(self, list sentences, list tags):
        """
        Trains for one epoch with all examples.
        """
        self.error = 0
        self.skips = 0
        self.float_errors = 0
        self.train_items = 0
        
        # shuffle data
        # get the random number generator state in order to shuffle
        # sentences and their tags in the same order
        random_state = np.random.get_state()
        np.random.shuffle(sentences)
        np.random.set_state(random_state)
        np.random.shuffle(tags)
        
        # keep last 2% for validation
        validation = int(len(sentences) * 0.98)
        i = 0
        for sent, sent_tags in izip(sentences, tags):
            try:
                self._tag_sentence(sent, True, sent_tags)
                self.train_items += len(sent)
            except FloatingPointError:
                # just ignore the sentence in case of an overflow
                self.float_errors += 1
            i += 1
            if i == validation:
                break
        self._validate(sentences, tags, validation)

    def _validate(self, sentences, tags, idx):
        """Perform validation on held out data and estimate accuracy"""
        tokens = 0
        hits = 0
        for i in xrange(idx, len(sentences)):
            sent = sentences[i]
            gold_tags = tags[i]
            scores = self._tag_sentence(sent, False)
            answer = self._viterbi(scores)
            for pred_tag, gold_tag in izip(answer, gold_tags):
                if pred_tag == gold_tag:
                    hits += 1
                tokens += 1
        self.accuracy = float(hits) / tokens

    
    def _backpropagate(self, sentence):
        """Backpropagate the error gradient."""
        # find the hidden gradients by backpropagating the output
        # gradients and multiplying the derivative
        # ((len(sentence), output_size)) x (output_size, hidden_size) = (len, hidden_size) Attardi
        cdef np.ndarray[FLOAT_t, ndim=2] hidden_gradients = self.net_gradients.dot(self.output_weights)
        
        # the derivative of tanh(x) is 1 - tanh^2(x)
        #cdef np.ndarray derivatives = 1 - self.hidden_sent_values ** 2
        # SENNA. Attardi
        # the derivative of htanh(x) is 1 if -1 < x < 1, else 0
        cdef np.ndarray derivatives = hardtanhd(self.hidden_sent_values)
        hidden_gradients *= derivatives
        
        # backpropagate to input layer (in order to adjust features)
        # since no function is applied to the feature values, no derivative is needed
        # (or you can see it as f(x) = x --> f'(x) = 1)
        # ((len(sentence), hidden_size) x (hidden_size, input_size) = (len, input_size) Attardi
        cdef np.ndarray[FLOAT_t, ndim=2] input_gradients = hidden_gradients.dot(self.hidden_weights)
        
        """
        Adjust the weights of the neural network.
        """
        # tensor[i, j, k] means the gradient for tag i at token j to be multiplied
        # by the value from the k-th hidden neuron (note that the tensor was transposed)
        cdef np.ndarray[FLOAT_t, ndim=3] grad_tensor
        
        # adjust weights from input to hidden layer
        grad_tensor = np.tile(hidden_gradients, [self.input_size, 1, 1]).T
        grad_tensor *= self.input_sent_values
        deltas = grad_tensor.sum(1) * self.learning_rate
        # Attardi: divide by the fan-in:
        #deltas = grad_tensor.sum(1) * (self.learning_rate / self.input_size)
        self.hidden_weights += deltas
        self.hidden_bias += hidden_gradients.sum(0) * self.learning_rate
        #self.hidden_bias += hidden_gradients.sum(0) * (self.learning_rate / self.input_size)
        
        # adjust weights from hidden to output layer
        grad_tensor = np.tile(self.net_gradients, [self.hidden_size, 1, 1]).T
        grad_tensor *= self.hidden_sent_values
        deltas = grad_tensor.sum(1) * self.learning_rate
        # Attardi: divide by the fan-in:
        #deltas = grad_tensor.sum(1) * (self.learning_rate / self.hidden_size)
        self.output_weights += deltas
        self.output_bias += self.net_gradients.sum(0) * self.learning_rate
        #self.output_bias += self.net_gradients.sum(0) * (self.learning_rate / self.hidden_size)
        
        """
        Adjust the features indexed by the input window.
        """
        # the deltas that will be applied to feature tables
        # they are in the same sequence as the network receives them, i.e.,
        # [token1-table1][token1-table2][token2-table1][token2-table2] (...)
        # (len(sentence), input_size). Attardi
        # input_size = num features * window (e.g. 60 * 5). Attardi
        cdef np.ndarray[FLOAT_t, ndim=2] input_deltas
        input_deltas = input_gradients * self.input_sent_values * self.learning_rate_features
        
        # this tracks where the deltas for the next table begins
        # (used for efficiency reasons)
        #cdef int start_from = 0
        cdef np.ndarray[FLOAT_t, ndim=2] table
        #cdef np.ndarray[INT_t, ndim=1] token
        cdef int i, j
        
        padded_sentence = np.concatenate((self.pre_padding,
                                          sentence,
                                          self.pos_padding))
        
        # for i in range(self.word_window_size):
        #     for j, table in enumerate(self.feature_tables):
        #         # this is the column for the i-th position in the window
        #         # regarding features from the j-th table
        #         table_deltas = input_deltas[:, start_from:start_from + table.shape[1]]
        #         start_from += table.shape[1]
        #         # token = [index in each feature_tables]. Attardi
        #         for token, deltas in izip(padded_sentence[i:], table_deltas):
        #             table[token[j]] += deltas

        cdef np.ndarray[INT_t, ndim=1] features
        cdef int start, end, t

        for i, w_deltas in enumerate(input_deltas):
            # for each window (w_deltas: 300, features: 5)
            start = 0
            for features in padded_sentence[i:i+self.word_window_size]:
                # select the columns for each feature_tables (t: 3)
                for t, table in enumerate(self.feature_tables):
                    end = start + table.shape[1]
                    table[features[t]] += w_deltas[start:end]
                    start = end

        # Adjusts the transition scores table with the calculated gradients.
        if self.transitions is not None:
            self.transitions += self.trans_gradients * self.learning_rate_trans

    def save(self, filename):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes, padding and 
        distance tables, but not other feature tables.
        """
        np.savez(filename, hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_bias=self.hidden_bias, output_bias=self.output_bias,
                 word_window_size=self.word_window_size, 
                 input_size=self.input_size, hidden_size=self.hidden_size,
                 output_size=self.output_size, padding_left=self.padding_left,
                 padding_right=self.padding_right, transitions=self.transitions)
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes, padding and 
        distance tables, but not other feature tables.
        """
        data = np.load(filename)
        
        # cython classes don't have the __dict__ attribute
        # so we can't do an elegant self.__dict__.update(data)
        hidden_weights = data['hidden_weights']
        hidden_bias = data['hidden_bias']
        output_weights = data['output_weights']
        output_bias = data['output_bias']
        
        word_window_size = data['word_window_size']
        input_size = data['input_size']
        hidden_size = data['hidden_size']
        output_size = data['output_size']
        
        nn = Network(word_window_size, input_size, hidden_size, output_size,
                     hidden_weights, hidden_bias, output_weights, output_bias)
        
        nn.padding_left = data['padding_left']
        nn.padding_right = data['padding_right']
        nn.pre_padding = np.array((nn.word_window_size / 2) * [nn.padding_left])
        nn.pos_padding = np.array((nn.word_window_size / 2) * [nn.padding_right])
        
        if 'transitions' in data:
            transitions = data['transitions']
            if transitions.shape != ():
                nn.transitions = transitions 
        
        return nn
        
# include the files for other networks
# this comes here after the Network class has already been defined
include "networkconv.pyx"
include "networklm.pyx"

