# Pos-Tag-By-NN-Daily-Star
Parts of speech tag by NN on Daily star Data

## WHAT?
A Part-Of-Speech Tagger (POS Tagger) is a piece of software that reads text in some language and assigns parts of speech to each word (and other token), such as noun, verb, adjective, etc.

## WHY?
Tag every word on sentence by a tag.

## HOW?
Word-Embedding model give us word-vector of every word on a sentence on a contexual manner. First generate Word-Embedding by using 4.0 GB Daily Star news content. Window size define as 300. 
Used 2 layer neural network with Adam optimizer for classifiy every words on sentence. CONLL taged sentenced used as a trained and validation data.

## THEN
Last night I lost the world, and gained the universe.” 
― C. JoyBell C.
