# SimpleRNN
An implementation of two simple Recurrent Neural Networks: Elman and Jordan nets for stocks prediction.
# Launching the program
To launch my interpretation of the Jordan and Elman nets follow these steps:

## Choosing a dataset.

These networks were made to use datasets of financial or any other type of data prediction. It's recommended the use of a
csv file with only one column per row, containing the ordered values for whatever you want to predict, with the
oldest value first and the newest last.

You can also use the data located in originalFiles/dat_entrada.csv in this repository, which contains financial data
for Iberdrola in between December 2016 and December 2017.

## Setting the values.

Once you have your dataset, you can open _main.m_ from the root of this repository on your Octave IDE and edit the variable
_ruta\_fichero_ to point to your data file.

There's a bunch more options you can set, like the hidden layer size (variable _capa\_oculta_), the input layer size (_capa\_entrada_),
the output layer size (_capa\_salida_), although it's recommended these two are left on their default value of 1,
since changing that might require tweaking with the implementations.

You can also change the learning rate (_factor\_aprendizaje_) and momentum rate (_factor\_inercia_) and the maximum of
epochs the nets will go through, _max\_epocas_. You should set _tam\_ventana_ to the size you think should be an appropriate
number of values to predict any new value. **The dataset size must be divisible by this number**.

## Launching the program.

After that simply write `main` on your Octave console and the program will start.

# Program's output.

The program will use the dataset to train and verify the quality of the Elman net first, and then plot the rate of success over time
graph.

It will do the same for the Jordan net, and then output its graph and another one comparing both nets.

Since this program is simply of investigative value, it doesn't output the weights of the trained nets, but a bit of tweaking can get
you these values so you can use them to predict new values!
