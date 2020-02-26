
# DELFT
The source codes of the paper "Complex Factoid Question Answering with a Free-Text Knowledge Graph (WWW 2020)".

# Dependency Installation
Run python setup.py develop to install required dependencies for DELFT.


# Data and trained model Download

For each experimented dataset, we provide processed graph input [here](https://obj.umiacs.umd.edu/delft_release/grounded_graph.zip), after downloading, unzip it and put into data folder (the tokenized version is for glove setting, while the other is for bert setting).
We also provide trained model [here](https://obj.umiacs.umd.edu/delft_release/trained_models.zip), unzip the downloaded model and put into experiments folder. Also, the original datasets are [here](https://obj.umiacs.umd.edu/delft_release/original_questions.zip).

For delft glove, downloading pre-trained glove embeddings are required, the link is [here](http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip).


## Run Experiments
The experiments include DELFT-Bert and DELFT-Glove , with different embeddings (Glove or Bert), each experiment has a seperate folder (see readme.md on each folder).


## Contact
If you have questions, suggestions and bug reports, please email chenz@cs.umd.edu.