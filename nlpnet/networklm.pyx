# -*- coding: utf-8 -*-

import utils

# batch size for traing the LanguageModel
# should be a class variable (not allowed in Cython)
cdef int RandomPool_size = 1000

cdef class RandomPool:

    cdef pool
    cdef tables
    cdef int current

    def __init__(self, tables):
        self.tables = tables
        self._new_pool()

    def _new_pool(self):            
        """
        Creates a pool of random feature indices, used for negative examples.
        """
        # generate 1000 indices for each table and then transpose
        # so that each row represents a token
        self.pool = np.array([np.random.random_integers(0, table.shape[0] - 1, RandomPool_size) 
                                    for table in self.tables], dtype=FLOAT).T
        self.current = 0

    def next(self):
        """
        Generates randomly a token for use as a negative example.
        :return: a list of token features, one for each feature table
        """
        if self.current == len(self.pool):
            self._new_pool()
        
        token = self.pool[self.current]
        self.current += 1
        
        return token


cdef class LanguageModel(Network): 
    """
    A neural network for the language modeling task, aimed 
    primiraly at inducing word representations.
    """
    
    # sizes and learning rates
    cdef int half_window
    
    # data for statistics during training. 
    cdef int total_items
    
    # pool of random numbers (used for efficiency)
    cdef RandomPool random_pool

    # file where to save model (Attardi)
    cdef public char* filename
    
    # learning rates for each layer
    cdef float LR_0, LR_1, LR_2

    @classmethod
    def create_new(cls, feature_tables, int word_window, int hidden_size):
        """
        Creates a new neural network initialized for training.
        """
        # sum the number of features in all tables 
        cdef int input_size = sum(table.shape[1] for table in feature_tables)
        input_size *= word_window
        
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh

        # creates the weight matrices
        #high = 2.38 / np.sqrt(input_size) # [Bottou-88]
        high = 2.45 / np.sqrt(input_size + hidden_size) # Al-Rfou
        hidden_weights = np.random.uniform(-high, high, (hidden_size, input_size))
        high = 2.38 / np.sqrt(hidden_size) # [Bottou-88]
        #hidden_bias = np.random.uniform(-high, high, (hidden_size))
        hidden_bias = np.zeros(hidden_size) # Al-Rfou

        high = 2.45 / np.sqrt(hidden_size + 1) # Al-Rfou
        output_weights = np.random.uniform(-high, high, (hidden_size))
        #high = 0.1
        #output_bias = np.random.uniform(-high, high, (1))
        output_bias = np.array([0.0], dtype=FLOAT) # Al-Rfou
        
        nn = cls(word_window, input_size, hidden_size, 
                 hidden_weights, hidden_bias, output_weights, output_bias)
        nn.feature_tables = feature_tables
        
        return nn
    
    def __init__(self, word_window, input_size, hidden_size, 
                 hidden_weights, hidden_bias, output_weights, output_bias):
        """
        This method shouldn't be called directly.
        Instead, use the classmethods load_from_file() or create_new().
        """
        # These will be set from arguments --l and --lf
        self.learning_rate = 0.1
        self.learning_rate_features = 0.1
        
        self.word_window_size = word_window
        self.half_window = self.word_window_size / 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias
        self.filename = ''      # Attardi
    
    
    def _train_pair(self, example):
        """
        Trains the network with a pair of positive/negative examples.
        The negative one is randomly generated.
	:param example: the positive example, i.e. a list of a list of token IDs
        """
        cdef np.ndarray[INT_t, ndim=1] token
        cdef int i, j
        cdef np.ndarray[FLOAT_t, ndim=2] table
        
        # a token is a list of feature IDs.
        # token[0] is the list with the WordDictionary index of the word
        middle_token = example[self.half_window]
        while True:
            # ensure to get a different word
            variant = self.random_pool.next()
            if variant[0] != middle_token[0]:
                break
        
        pos_input_values = self.lookup(example)
        pos_score = self.run(pos_input_values)
        pos_hidden_values = self.hidden_values
        
        negative_token = np.array(variant, dtype=FLOAT)
        example[self.half_window] = negative_token
        neg_input_values = self.lookup(example)
        neg_score = self.run(neg_input_values)
        
        # hinge loss
        error = max(0, 1 - pos_score + neg_score)
        self.error += error
        self.total_items += 1
        if error == 0: 
            self.skips += 1
            return
        
        # perform the correction
        # negative gradient for the positive example is +1, for the negative one is -1
        # (remember the network still has the values of the negative example) 
        
        # output gradients
        # pos_output_grads = 1
        # neg_output_grads = -1
        # (output_size) x (output_size, hidden_size) = (hidden_size)
        # pos_output_weights_grads = pos_output_grads * self.output_weights
        # neg_output_weights_grads = neg_output_grads * self.output_weights

        # hidden gradients
        # (hidden_size) * (hidden_size) = (hidden_size)
        layer2_neg_grads = hardtanhe(self.hidden_values) * (- self.output_weights)
        layer2_pos_grads = hardtanhe(pos_hidden_values) * self.output_weights
        
        # input gradients
        # (hidden_size) x (hidden_size, input_size) = (input_size)
        input_neg_grads = self.LR_0 * layer2_neg_grads.dot(self.hidden_weights)
        input_pos_grads = self.LR_0 * layer2_pos_grads.dot(self.hidden_weights)
        
        # weight adjustment
        # output bias is left unchanged -- a correction would imply in bias += +delta -delta

        # (output_size) x (hidden_size) = (output_size, hidden_size)
        # (1) x (hidden_size) = (1, hidden_size)
        self.output_weights += self.LR_2 * (pos_hidden_values - self.hidden_values) 
        
        # hidden weights
        # (hidden_size) x (input_size) = (hidden_size, input_size)
        hidden_neg_grads = np.outer(layer2_neg_grads, neg_input_values)
        hidden_pos_grads = np.outer(layer2_pos_grads, pos_input_values)
        
        self.hidden_weights += self.LR_1 * (hidden_neg_grads + hidden_pos_grads)
        self.hidden_bias += self.LR_1 * (layer2_neg_grads + layer2_pos_grads)
        
        # this tracks where the deltas for the next table begins
        cdef int offset = 0
             
        for i, token in enumerate(example):
            for j, table in enumerate(self.feature_tables):
                # this is the column for the i-th position in the window
                # regarding features from the j-th table
                neg_grads = input_neg_grads[offset: offset + table.shape[1]]
                pos_grads = input_pos_grads[offset: offset + table.shape[1]]
                if i == self.half_window:
                    # this is the middle position. apply negative and positive deltas separately
                    
                    table[negative_token[j]] += neg_grads
                    table[middle_token[j]] += pos_grads
                else:
                    # this is not the middle position. both deltas apply.
                    table[token[j]] += neg_grads + pos_grads
                
                offset += table.shape[1]

                
    def train(self, list sentences, int epochs, int iterations_between_reports):
        """
        Trains the language model over the given sentences.
        :param epochs: number of iterations over the sentences
        """
        
        # generate 1000 random indices at a time to save time
        # (generating 1000 integers at once takes about ten times the time for a single one)
        self.random_pool = RandomPool(self.feature_tables)
        self.total_items = 0
        
        # how often to save model
        save_period = 1000 * RandomPool_size

        all_cases = sum([len(sen) for sen in sentences]) * epochs

        # TODO: parallelize
        # see: http://radimrehurek.com/2013/10/parallelizing-word2vec-in-python/
        # all threads access the same matrix of neural weights, and there’s no
        # locking, so they may be overwriting the same weights willy-nilly at
        # the same time. Apparently this hack even has a fancy name in
        # academia: “asynchronous stochastic gradient descent”.

#         from queue import Queue
#
#         jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs
#         lock = threading.Lock()  # for shared state
#
#         def worker_train():
#             """Train the model, lifting lists of sentences from the jobs queue."""
#             params = Params()  # each thread has its own hidden weights
#             gradients = Params()  # and gradients
#
#             while True:
#                 job = jobs.get()
#                 if job is None:  # data finished, exit
#                     break
#                 job_words = self._train_batch(self, params, gradients)
#                 with lock:
#                     word_count[0] += job_words
#                     elapsed = time.time() - start
#                     if elapsed >= next_report[0]:
#                         logger.info("PROGRESS: at %.2f%% examples" %
#                             (100.0 * word_count[0] / total_words))
#                         # wait at least one second between progress reports
#                         next_report[0] = elapsed + 1.0
#         workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
#         for thread in workers:
#             thread.daemon = True  # make interrupting the process with ctrl+c easier
#             thread.start()
#
#         # fill the job queue
#         for job in grouper(sentences, chunksize):
#             jobs.put(job)
#         for _ in xrange(self.workers):
#             jobs.put(None)  # signal finish
#
#         for thread in workers:
#             thread.join()

        for epoch in xrange(epochs):
            self.error = <FLOAT_t>0.0
            self.skips = 0
            epoch_examples = 0
            # update LR by fan-in
            # decrease linearly by remaining
            remaining = 1.0 - (self.total_items / float(all_cases))
            self.LR_0 = max(0.001, self.learning_rate * remaining)
            self.LR_1 = max(0.001, self.learning_rate / self.input_size * remaining)
            self.LR_2 = max(0.001, self.learning_rate / self.hidden_size * remaining)

            for sentence in sentences:
                for pos in xrange(len(sentence)):
                    # extract the window around the given position
                    window = self._extract_window(sentence, pos)
                
                    self._train_pair(window)
                    epoch_examples += 1

                    if iterations_between_reports > 0 and \
                       (self.total_items and
                        self.total_items % iterations_between_reports == 0):
                        self._progress_report(epoch, epoch_examples)
                        # save language model. Attardi
                        if save_period and self.total_items % save_period == 0:
                            utils.save_features_to_file(self.feature_tables[0], self.filename)
    
    def _extract_window(self, sentence, position):
        """
        Extracts a token window from the sentence, with the size equal to the
        network's window size.
        This function takes care of creating padding as necessary.
        """
        if position < self.half_window:
            num_padding = self.half_window - position
            pre_padding = np.array(num_padding * [self.padding_left], dtype=np.int)
            sentence = np.concatenate((pre_padding, sentence))
            position += num_padding
        
        # number of tokens in the sentence after the position
        tokens_after = len(sentence) - position - 1
        if tokens_after < self.half_window:
            num_padding = self.half_window - tokens_after
            pos_padding = np.array(num_padding * [self.padding_right], dtype=np.int)
            sentence = np.concatenate((sentence, pos_padding))
        
        return sentence[position - self.half_window : position + self.half_window + 1]
    
    def description(self):
        """
        Returns a description of the network.
        """
        table_dims = [str(t.shape[1]) for t in self.feature_tables]
        table_dims =  ', '.join(table_dims)
        
        desc = """
Word window size: %d
Feature table sizes: %s
Input layer size: %d
Hidden layer size: %d
""" % (self.word_window_size, table_dims, self.input_size, self.hidden_size)
        
        return desc
    
    def save(self, filename):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes, and padding,
        but not the feature tables nor the vocabulary.
        """
        np.savez(filename, hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_bias=self.hidden_bias, output_bias=self.output_bias,
                 word_window_size=self.word_window_size, 
                 input_size=self.input_size, hidden_size=self.hidden_size,
                 padding_left=self.padding_left, padding_right=self.padding_right)
    
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes and padding, but 
        not feature tables.
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
        
        nn = cls(word_window_size, input_size, hidden_size, 
                 hidden_weights, hidden_bias, output_weights, output_bias)
        
        nn.padding_left = data['padding_left']
        nn.padding_right = data['padding_right']
        
        return nn
    
    def _progress_report(self, int num, epoch_examples):
        """
        Reports the status of the network in the given training
        epoch, including error and accuracy.
        """
        # FIXME: should use moving average
        cdef float error = self.error / self.total_items
        logger = logging.getLogger("Logger")
        logger.info("Epoch %d, examples: %d, correct: %d/%d, error: %.3f%%"
                    % (num + 1, self.total_items, self.skips, epoch_examples, error * 100))

