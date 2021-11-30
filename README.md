# Beethoven Piano Music Creator

Original repository: https://github.com/Skuldur/Classical-Piano-Composer by Skuldur

The original network was trained on Final Fantasy video game music. The same RNN network was trained from scratch on Beethoven's classical piano pieces to generate new 2 minute long compostions. 

Trainibg loss > 0.5 leads to underfitting in which case the same note is repeated indefinitely. 

Training loss < 0.2 leads to overfitting where one of the midi tracks is replicated with little to no changes. 

Beethoven Dataset: http://www.piano-midi.de/beeth.htm

Download weights file (loss 0.3593 on above dataset): https://drive.google.com/open?id=1023aIYiubcYL1hvmM7T2xfNEprMvCoSy

## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21
	* Keras
	* Tensorflow
	* h5py

## Training

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
