# Sequence-to-Sequence-Models

## I implement a simple character level Sequence to Sequence Model using Keras, following the Tensorflor tutorials: https://www.tensorflow.org/tutorials/sequences/text_generation. I use the model to generate new song lyrics given a starting word or phrase. 

### In the tutorial the dataset used is the dataset of Shakespeare's work which follows the blog by Andrej Karpathy http://karpathy.github.io/2015/05/21/rnn-effectiveness/. I use my own dataset which consists of the dataset of 380000+ songs downloaded from Kaggle here: https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics. 

### to reduce the dataset size, I specifically limit to a single genre "indie" in this case, which reduces the dataset to about 5700 songs. Even this dataset contains over 3 million characters compared to the 1 million characters needed to train the Shakespeare dataset. 

### Extract_Song_lyrics.py extracts the relevant lyrics from the database consisting of indie songs, and after some data cleaning and removal of unwanted characters which are common when transcribing songs,but don't actually add any information to the model, I am left with only text containing alphabetical characters, newline character and a few additional punctuations. The main.py file then takes this dataset as input and builds a dict mapping characters to numbers. 

### The lyrics are broken up into sequences of fixed length. To train the model we feed the model a sequence and have it predict the same sequence shifted by one character. The model itself consists of three parts:
  # 1) an embedding layer which maps the sequence vector to a tensor in a 256 dimensional space producing an output of [batch_size, sequence_length, embedding vector]. The vectors representations are learned during training. 
  # 2) a layer of GRUs which do the heavy lifting and convert the embedding tensors into hidden representations. Unrolling the model through time, at each time step the model uses the hidden state from the previous GRU layer in addition to the character representation to learn the context in which the characters appear and learn words.
  # 3) A final dense layer which maps the output of the GRU back into the vocab vectors to yield the logits - which is the probabilty of predicting the next character. 

### An Adam optimizer and cross entropy loss is used to train the weights. The model is trained on Google Colab and runs pretty quickly when the GPU is enabled. 

### Improvements: As you can see, sometimes the lyrics make sense, other times they don't. As the prediction is probabilistic, new lyrics are generated for the same character. The model however is quite simplistic and trains at a character level. More advanced models train at a word or sentence level. Watch out for implementations of these! 

