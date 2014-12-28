# -*- coding: utf-8 -*- 

cdef class SentimentModel(LanguageModel): 
    """
    A neural network for sentiment specific language model, aimed 
    at inducing sentiment-specific word representations.
    @see Tang et al. 2014. Learning Sentiment-SpecificWord Embedding for Twitter Sentiment Classification.
    http://aclweb.org/anthology/P14-1146
    """
    
    # polarities of each tweet
    cdef list polarities

    # alpha parameter
    cdef double alpha

    cdef np.ndarray neg_hidden_adagrads
    cdef np.ndarray pos_hidden_adagrads

    @classmethod
    def create_new(cls, feature_tables, int word_window, int hidden_size, float alpha):
        """
        Initializes a new neural network initialized for training.
        :param word_window: defaut 3
        :param hidden_size: default 20
        :param alpha: default 0.5
        """
        # sum the number of features in all tables 
        cdef int input_size = sum(table.shape[1] for table in feature_tables)
        input_size *= word_window
        
        # creates the weight matrices
        #high = 2.38 / np.sqrt(input_size) # [Bottou-88]
        high = 2.45 / np.sqrt(input_size + hidden_size) # Al-Rfou
        hidden_weights = np.random.uniform(-high, high, (hidden_size, input_size))
        high = 2.38 / np.sqrt(hidden_size) # [Bottou-88]
        #hidden_bias = np.random.uniform(-high, high, (hidden_size))
        hidden_bias = np.zeros(hidden_size, dtype=FLOAT) # Al-Rfou

        # There are two output weights: syntactic and sentiment
        high = 2.45 / np.sqrt(hidden_size + 2) # Al-Rfou
        output_weights = np.random.uniform(-high, high, (2, hidden_size))
        #output_bias = np.random.uniform(-high, high, (2))
        output_bias = np.array([0.0, 0.0])       # Al-Rfou
        
        nn = SentimentModel(word_window, input_size, hidden_size, 
                            hidden_weights, hidden_bias, output_weights, output_bias)
        nn.feature_tables = feature_tables
        nn.alpha = alpha

        # cumulative AdaGrad
        nn.neg_hidden_adagrads = np.zeros(hidden_size, dtype=FLOAT)
        nn.pos_hidden_adagrads = np.zeros(hidden_size, dtype=FLOAT)

        return nn

    def __init__(self, word_window, input_size, hidden_size, 
                 hidden_weights, hidden_bias, output_weights, output_bias):
        """
        This method shouldn't be called directly.
        Instead, use the classmethods load_from_file() or create_new().
        """
        LanguageModel.__init__(self, word_window, input_size, hidden_size,
                               hidden_weights, hidden_bias,
                               output_weights, output_bias)

    def _train_pair(self, example, polarity, size):
        """
        Trains the network with a pair of positive/negative examples.
        The negative one is randomly generated.
	:param example: the positive example, i.e. a list of a list of token IDs
        :param polarity: 1 for positive, -1 for negative sentences.
	:param size: size of ngram to generate for replacing window center
        """
        cdef np.ndarray[INT_t, ndim=1] token
        cdef int i, j
        cdef np.ndarray[FLOAT_t, ndim=2] table
        
        # a token is a list of feature IDs.
        # token[0] is the list with the WordDictionary index of the word
        middle_token = example[self.half_window]

        if size == 1:
	   # ensure to generate a different word
            while True:
                variant = self._generate_token()
                if variant[0] != middle_token[0]:
                    break

        pos_input_values = self.lookup(example)
        pos_score = self.run(pos_input_values)
        pos_hidden_values = self.hidden_values
        
        negative_token = np.array(variant)
        example[self.half_window] = negative_token
        neg_input_values = self.lookup(example)
        neg_score = self.run(neg_input_values)
        
        errorCW = max(0, 1 - pos_score[0] + neg_score[0])
        errorUS = max(0, 1 - polarity * pos_score[1] + polarity * neg_score[1])
        error = self.alpha * errorCW + (1 - self.alpha) * errorUS
        self.error += error
        self.total_items += 1
        if error == 0: 
            self.skips += 1
            return
        
        # perform the correction
        # (remember the network still has the values of the negative example) 

        # negative gradient for the positive example is +1, for the negative one is -1
        # @see A.8 in Collobert et al. 2011.
        pos_score_grads = np.array([0, 0])
        neg_score_grads = np.array([0, 0])
        if (errorCW > 0):
            pos_score_grads[0] = 1
            neg_score_grads[0] = -1
        if (errorUS > 0):
            pos_score_grads[1] = 1
            neg_score_grads[1] = -1
        
        # Summary:
        # output_bias_grads = score_grads
        # output_weights_grads = score_grads.T * hidden_values
        # hidden_grads = activationError(hidden_values) * score_grads.T.dot(output_weights)
        # hidden_bias_grads = hidden_grads
        # hidden_weights_grads = hidden_grads.T * input_values
        # input_grads = hidden_grads.dot(hidden_weights)

        # Output layer
        # CHECKME: summing they cancel each other:
        cdef np.ndarray output_bias_grads = pos_score_grads + neg_score_grads
        # (2) x (hidden_size) = (2, hidden_size)
        cdef np.ndarray output_weights_grads = np.outer(pos_score_grads, pos_hidden_values) + np.outer(neg_score_grads, self.hidden_values)

        # Hidden layer
        # (2) x (2, hidden_size) = (hidden_size)
        neg_hidden_grads = hardtanhe(self.hidden_values) * neg_score_grads.dot(self.output_weights)
        pos_hidden_grads = hardtanhe(pos_hidden_values) * pos_score_grads.dot(self.output_weights)

        # Input layer
        # (hidden_size) x (input_size) = (hidden_size, input_size)
        cdef np.ndarray neg_hidden_weights_grads = np.outer(neg_hidden_grads, neg_input_values)
        cdef np.ndarray pos_hidden_weights_grads = np.outer(pos_hidden_grads, pos_input_values)
        cdef np.ndarray hidden_weights_grads = pos_hidden_weights_grads + neg_hidden_weights_grads
        cdef np.ndarray hidden_bias_grads = pos_hidden_grads + neg_hidden_grads

        # weight adjustment
        self.output_weights += self.LR_2 * output_weights_grads
        self.output_bias += self.LR_2 * output_bias_grads
        
        self.hidden_weights += self.LR_1 * hidden_weights_grads
        self.hidden_bias += self.LR_1 * hidden_bias_grads
        
        # input gradients, using AdaGrad
        self.neg_hidden_adagrads += np.power(neg_hidden_grads, 2)
	# (hidden_size) x (hidden_size, input_size) = (input_size)
        neg_input_grads = (neg_hidden_grads / np.sqrt(self.neg_hidden_adagrads)).dot(self.hidden_weights)

        self.pos_hidden_adagrads += np.power(pos_hidden_grads, 2)
        pos_input_grads = (pos_hidden_grads / np.sqrt(self.pos_hidden_adagrads)).dot(self.hidden_weights)

        neg_input_deltas = self.LR_0 * neg_input_grads
        pos_input_deltas = self.LR_0 * pos_input_grads
        
        # this tracks where the deltas for the next table begins
        cdef int offset = 0
             
        for i, token in enumerate(example):
            for j, table in enumerate(self.feature_tables): # just one table
                # this is the column for the i-th position in the window
                # regarding features from the j-th table
                neg_deltas = neg_input_deltas[offset: offset + table.shape[1]]
                pos_deltas = pos_input_deltas[offset: offset + table.shape[1]]
                    
                if i == self.half_window:
                    # this is the middle position.
                    # apply negative and positive deltas to proper token
                    table[negative_token[j]] += neg_deltas
                    table[middle_token[j]] += pos_deltas
                else:
                    # this is not the middle position. both deltas apply.
                    table[token[j]] += neg_deltas + pos_deltas
                
                offset += table.shape[1]
    
    def train(self, list sentences, int epochs, int iterations_between_reports, list polarities, ngram_dict):
        """
        Trains the sentiment language model on the given sentences.
        :param sentences: list of token IDs for each sentence
        :param iterations: number of train iterations
        :param polarities: the polarity of each sentence, +-1.
        :param ngram_dixt: the dictionary of the ngrams on the corpus
        """
        # generate 1000 random indices at a time to save time
        # (generating 1000 integers at once takes about ten times the time for a single one)
        self.random_pool = RandomPool(self.feature_tables)
        self.total_items = 0

        # how often to save model
        save_period = 1000 * RandomPool_size

        all_cases = sum([len(sen) for sen in sentences]) * epochs * self.ngrams

        for epoch in xrange(self.epochs):
            self.error = 0.0
            self.skips = 0
            epoch_examples = 0
            # update LR by fan-in
            # decrease linearly by remaining
            remaining = 1.0 - (self.total_items / float(all_cases))
            self.LR_0 = max(0.001, self.learning_rate * remaining)
            self.LR_1 = max(0.001, self.learning_rate / self.input_size * remaining)
            self.LR_2 = max(0.001, self.learning_rate / self.hidden_size * remaining)

            for num, sentence in enumerate(sentences):
                for pos in xrange(len(sentence)):
                
                    # ngram size changes periodically
                    if self.total_items:
                        if self.total_items % 5 == 0:
                            size = 2
                        elif self.total_items % 17 == 0:
                            size = 3
                        else:
                            size = 1
                    else:
                        size = 1

                    # extract a window of tokens around the given position
                    window = self._extract_window(sentence, pos, size, ngram_dict)

                    self._train_pair(window, polarities[num], size)
                    epoch_examples += 1

                    if iterations_between_reports > 0 and \
                       (self.total_items and
                        self.total_items % iterations_between_reports == 0):
                        self._progress_report(epoch, epoch_examples)
                        # save language model. Attardi
                        if save_period and self.total_items % save_period == 0:
                            utils.save_features_to_file(self.feature_tables[0], self.filename)
    
    def _extract_window(self, sentence, position, size=1, ngram_dict=None):
        """
        Extracts a window of tokens from the sentence, with size equal to
        the network's window size.
        This function takes care of creating padding as necessary.
	:param sentence: the sentence from which to extract the window
	:param position: the center token position
        :param size: the size of ngram in the center of thw window
	:return: a portion of sentence centered at position
        """
        if position < self.half_window:
            num_padding = self.half_window - position
            pre_padding = np.array(num_padding * [self.padding_left])
            sentence = np.vstack((pre_padding, sentence))
            position += num_padding
        
        # number of tokens in the sentence after the position
        tokens_after = len(sentence) - (position + size)
        if tokens_after < self.half_window:
            num_padding = self.half_window - tokens_after
            pos_padding = np.array(num_padding * [self.padding_right])
            sentence = np.vstack((sentence, pos_padding))
        
        ngram = sentence[position: position + size]
        if size > 1:
            # lookup ngram index
            tokens = ngram_dict.get_words(ngram)
            ngram = [[ngram_dict[' '.join(tokens)]]] # one feature_table

        return np.concatenate((sentence[position - self.half_window: position], ngram, sentence[position + size: position + size + self.half_window]))
    
    @classmethod
    def load_from_file(cls, filename):
        # inherit from base class
        return LanguageModel.load_from_file.__func__(cls, filename)
