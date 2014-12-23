===============================================================
``nlpnet`` --- Natural Language Processing with neural networks
===============================================================

Command line scripts
~~~~~~~~~~~~~~~~~~~~


Training
~~~~~~~~

The script ``nlpnet-train.py`` can be used to train a model for a specified task from an annotated corpus.

Specific configuration parameters are provided in the file ``nlpnet/config.py``.

The script is invoked as follows::

 usage: nlpnet-train.py [-h] --task {lm,ner,pos,srl,sslm} --gold GOLD --data DATA
                       [-w WINDOW] [-f NUM_FEATURES] [--load_features]
                       [-e ITERATIONS] [-l LEARNING_RATE]
                       [--lf LEARNING_RATE_FEATURES]
                       [--lt LEARNING_RATE_TRANSITIONS] [--caps [CAPS]]
                       [--suffix [SUFFIX]] [--pos [POS]] [--chunk [CHUNK]]
                       [--use_lemma] [--gazetteer [GAZETTEER]] [-a ACCURACY]
                       [-n HIDDEN] [-c CONVOLUTION] [-v] [--id] [--class]
                       [--pred] [--load_network] [--max_dist MAX_DIST]
                       [--target_features TARGET_FEATURES]
                       [--pred_features PRED_FEATURES]
                       [--semi SEMI] [--variant VARIANT]

 optional arguments:
  -h, --help            show this help message and exit
  --task {lm,ner,pos,srl,sslm}
                        Task for which the network should be used.
  --gold GOLD           File with annotated data for training.
  --data DATA           Directory to save new models and load partially
                        trained ones
  -w WINDOW, --window WINDOW
                        Size of the word window
  -f NUM_FEATURES, --num_features NUM_FEATURES
                        Number of features per word
  --load_features       Load previously saved word type features (overrides -f
                        and must also load a dictionary file)
  -e ITERATIONS, --epochs ITERATIONS
                        Number of training epochs
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for network connections (default 0.001)
  --lf LEARNING_RATE_FEATURES
                        Learning rate for features (default 0.01)
  --lt LEARNING_RATE_TRANSITIONS
                        Learning rate for transitions (default 0.01)
  --caps [CAPS]         Include capitalization features. Optionally, supply
                        the number of features (default 5)
  --suffix [SUFFIX]     Include suffix features. Optionally, supply the number
                        of features (default 5)
  --pos [POS]           Include part-of-speech features (for SRL only).
                        Optionally, supply the number of features (default 5)
  --chunk [CHUNK]       Include chunk features (for SRL only). Optionally,
                        supply the number of features (default 5)
  --use_lemma           Use word lemmas instead of surface forms.
  --gazetteer [GAZETTEER]
                        Include gazetteer features (for NER only). Optionally,
                        supply the number of features (default 5)
  -a ACCURACY, --accuracy ACCURACY
                        Desired accuracy per tag.
  -n HIDDEN, --hidden HIDDEN
                        Number of hidden neurons
  -c CONVOLUTION, --convolution CONVOLUTION
                        Number of convolution neurons
  -v, --verbose         Verbose mode
  --id                  Identify argument boundaries (do not classify)
  --class               Classify previously delimited SRL arguments
  --pred                Only predicate identification (SRL only)
  --load_network        Load previously saved network
  --max_dist MAX_DIST   Maximum distance to have a separate feature (SRL only)
  --target_features TARGET_FEATURES
                        Number of features for distance to target word (SRL
                        only)
  --pred_features PRED_FEATURES
                        Number of features for distance to predicate (SRL
                        only)
  --semi SEMI           Perform semi-supervised training. Supply the name of
                        the file with automatically tagged data.
  --variant VARIANT     If "polyglot" use Polyglot case conventions;
                        if "senna" use SENNA conventions.


Tagging
~~~~~~~

The script ``nlpnet-tag.py`` can be used for tagging.
It reads from standard input and can be invoked as follows::

 usage: nlpnet-tag.py [-h] [-v] [--no-repeat] {srl,pos,ner} data

 positional arguments:
  {srl,pos,ner}  Task for which the network should be used.
  data           Directory containing trained models.

 optional arguments:
  -h, --help     show this help message and exit
  -v             Verbose mode
  --no-repeat    Forces the classification step to avoid repeated argument
                 labels (SRL only).
