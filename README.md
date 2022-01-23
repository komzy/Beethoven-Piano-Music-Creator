# A Shallow Neural Network for Beethoven Piano Music Generation

Original repository: https://github.com/Skuldur/Classical-Piano-Composer by Skuldur

The original network was trained on Final Fantasy video game music. The same RNN network was trained from scratch on Beethoven's classical piano pieces to generate new 2 minute long compostions containing 500 notes. Musical notes are converted into integers and sequences of the noted(integers) are fed to the LSTM for training.

To generate music accurately our neural network will have to be able to predict which note or chord is next. That means that our prediction array will have to contain every note and chord object that we encounter in our training set.
Training loss > 0.5 leads to underfitting in which case the same note is repeated indefinitely. 

Training loss < 0.2 leads to overfitting where one of the midi tracks is replicated with little to no changes. 

Beethoven Dataset: http://www.piano-midi.de/beeth.htm

Download weights file (loss 0.3593 on above dataset): https://drive.google.com/open?id=1023aIYiubcYL1hvmM7T2xfNEprMvCoSy

## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21: For interpreting music data. Returns the array of notes and chords present in the musical file. Get unique notes(frequencies) and their distribution.
	* Keras
	* Tensorflow
	* h5py

## Network
3 x LSTM layers, 3 x Dropout layers, 2 x Dense layers and 1 x activation layer. 

## Hyperparameters
Categorical cross entropy is used since each of our outputs (notes/chord) only belongs to a single class and we have more than two classes to work with. RMSprop optimizer is used as it is usually a very good choice for recurrent neural networks.

## Training

The input is th note or chord as well as a sequence of 99 notes/chords previous to it (aka the context), mapped to one hot encoded integer values.
One node in the output layer represents a unique chord or note. 

The output for each input sequence will be the first note or chord that comes after the sequence of notes in the input sequence in our list of notes.
This means that to predict the next note in the sequence the network has the previous 100 notes to help make the prediction. 


To train the network you run **lstm.py**.

E.g.

```
python lstm3.py
```

The network will use every midi file in ./midi_songs to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

## Generating music


Once you have trained the network you can generate text using **predict.py**

E.g.

```
python predict3.py
```

You can run the prediction file right away using the **weights.hdf5** file

For each new note to be generate a sequence has to be submitted. We submit a first sequence as input and for every subsequent sequence the network uses as input it will remove the first note of the sequence and insert the output of the previous iteration at the end of the sequence. Then we collect all the outputs from the network into a single array.

## Limitations
The implementation supports only ONE instrument and does not support varying duration of notes and different offsets between notes. The can be seen as another class of input aside from notes and chords. There is also no method to handle unknown notes.
