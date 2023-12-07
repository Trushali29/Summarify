# Summarify
Summarify is a text summarization application. It is a NLP based project which aim to provide summaries of the documents such as news. 

# Steps to run the project
1. In install pre-trained GLoVe model from site : https://nlp.stanford.edu/projects/glove/ - Glove.6B.zip
2. Install pandas, nltk, spacy, CustomTkinter, numpy modules in using commmand - pip install modulename.
3. Get the BBC News summaries whle dataset from https://huggingface.co/datasets/gopalkalpande/bbc-news-summary
4. Run the bbc-news-dataset python file which will generate folders. Main folders are Articles and Summaries each consists tech, sport, business, politics and entertainment as folder.
5. Each 5 folders consist around 300 news articles.
6. Create generate summaries folder in bbc news dataset folder.
7. Run the main file.


# The Summarify application

![image](https://github.com/Trushali29/Summarify/assets/84562990/af4ea2a7-a5d5-4587-8581-522553f75820)

**See the running application video below** 


[Final summarify.webm](https://github.com/Trushali29/Summarify/assets/84562990/4815d8b8-007f-45c9-814f-4f2b14c5964e)

 
# Sample output of the entertainment article

## Article	
A Christmas tree that can receive text messages has been unveiled at London's Tate Britain art gallery.
 The spruce has an antenna which can receive Bluetooth texts sent by visitors to the Tate. The messages will be "unwrapped" by sculptor Richard Wentworth, who is responsible for decorating the tree with broken plates and light bulbs. It is the 17th year that the gallery has invited an artist to dress their Christmas tree. Artists who have decorated the Tate tree in previous years include Tracey Emin in 2002.
 
 The plain green Norway spruce is displayed in the gallery's foyer. Its light bulb adornments are dimmed, ordinary domestic ones joined together with string. The plates decorating the branches will be auctioned off for the children's charity ArtWorks. Wentworth worked as an assistant to sculptor Henry Moore in the late 1960s. His reputation as a sculptor grew in the 1980s, while he has been one of the most influential teachers during the last two decades. Wentworth is also known for his photography of mundane, everyday subjects such as a cigarette packet jammed under the wonky leg of a table.
 
## Original Summary	
The messages will be "unwrapped" by sculptor Richard Wentworth, who is responsible for decorating the tree with broken plates and light bulbs. A Christmas tree that can receive text messages has been unveiled at London's Tate Britain art gallery. It is the 17th year that the gallery has invited an artist to dress their Christmas tree. The spruce has an antenna which can receive Bluetooth texts sent by visitors to the Tate. His reputation as a sculptor grew in the 1980s, while he has been one of the most influential teachers during the last two decades.

## TF_IDF Summary	

A Christmas tree that can receive text messages has been unveiled at London's Tate Britain art gallery. The spruce has an antenna which can receive Bluetooth texts sent by visitors to the Tate. The messages will be "unwrapped" by sculptor Richard Wentworth, who is responsible for decorating the tree with broken plates and light bulbs. It is the 17th year that the gallery has invited an artist to dress their Christmas tree. Artists who have decorated the Tate tree in previous years include Tracey Emin in 2002.

## GloVe Summary	
A Christmas tree that can receive text messages has been unveiled at London's Tate Britain art gallery.
The messages will be "unwrapped" by sculptor Richard Wentworth, who is responsible for decorating the tree with broken plates and light bulbs.
It is the 17th year that the gallery has invited an artist to dress their Christmas tree.
Artists who have decorated the Tate tree in previous years include Tracey Emin in 2002.
His reputation as a sculptor grew in the 1980s, while he has been one of the most influential teachers during the last two decades.


## Text Rank Summary	
A Christmas tree that can receive text messages has been unveiled at London's Tate Britain art gallery. The spruce has an antenna which can receive Bluetooth texts sent by visitors to the Tate. The messages will be "unwrapped" by sculptor Richard Wentworth, who is responsible for decorating the tree with broken plates and light bulbs. It is the 17th year that the gallery has invited an artist to dress their Christmas tree. Artists who have decorated the Tate tree in previous years include Tracey Emin in 2002.
