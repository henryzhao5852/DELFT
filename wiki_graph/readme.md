# DELFT Wiki graph

We provide two versions of Wikipedia graph, using orginal links [here](https://obj.umiacs.umd.edu/delft_wiki_graph/enwiki_links_anchor_1101.zip) and using tagme entities [here](https://obj.umiacs.umd.edu/delft_wiki_graph/wiki_anchor_1101.zip).


# Data format


Each data file contains multiple json objects, and each object contains the following items:

* Page id: the Wikipedia id of the page
* Title: the Wikipedia title of the page
* Text: Sentence
* Anchored et : the tagged entities from Tagme or linked entities
    - Each tagme entity has the following: (Wikipedia title, entity start index, entity end index, tagme score, Wikipedia page id, surface form)
    - Each linked entity has the following: (Wikipedia title, surface form)


