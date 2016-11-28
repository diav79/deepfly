#DeepFly: Training ResNet for Multi Variate Regression, from Scratch, Step-by-Step

Prerequisite: Caffe (caffe.berkeleyvision.org/)

This code was used in DeepFly: Towards Complete Autonomous Navigation of MAVs with Monocular Camera, Utsav et al, ICVGIP 2016

##Data Preparation

Create a txt file that contains the path to the image and the multiple target values for regression; all separated by commas. For example, see train_list.txt; it has only two entries just to give idea.

#####This code assumes that there are four target values we are regressing, but this number can be easily changed (or turned into classification problem) by modifying some small parts as we shall see later.

Another important thing to note is that it is assumed that you are performing all the preprocessing beforehand. That is, your set of training images contain both original plus augmented set. This assumption is in place primarily because modifying image can change target value of variable. If the targets are not affected by any preprocessing (or you are performing classification), then you can specify those augmentation parameters in data layer of prototxt files.

Once the train_list.txt is ready with image path and target values, it is good idea to shuffle the data. Run 'python shuffle.py'. This will generate a file named "trainingData.txt". This file contains same entries as train_list.txt but in random order. Now, move appropriate number of entries from this file and save them in "testingData.txt"; this will be used for validation purposes.

Once you have trainingData.txt and testingData.txt populated with training and validation entries, run createLMDB.py to generate the lmdb. This can take long time if your database is big. After completion, you will find two new folders created that has lmdb for our task. If your training data is composed of grayscale images, then at line 72 of createLMDB.py in load_image(), change color=False.

Last step before starting training is to compute the image mean of the dataset by running ./compute_image_mean in caffe/tools/ and passing it the training lmdb as argument.

##Training

Set the paths of training and testing lmdbs and image mean in DiavNet_1.prototxt. Change the batch-size as per your GPU capacity. Make similar changes in DiavNet_Deploy_Res_1.prototxt too. Finally, change the parameters in DiavNet_1_Solver.prototxt as per your problem.

Run
path-to-caffe/build/tools/caffe train --solver=path-to/DiavNet_1_Solver.prototxt

That's it!

##Testing

Create a txt file that has image path per line. Change createLMDB.py -- change the list ['trainingData','testingData'] to ['yourNewTestingData']. Comment out the code which works on labels and generate lmdb for only images. After that run caffe test and pass model and learned weights as input args.

##Multi Variate Regression

Here, we are regressing four target variables. If you have more or less variables, change the lines 36 and 56 of createLMDB accordingly. If you want to perform single variable regression, toggle the comments of line 36-37 and 56-57. After that change the output neurons from 4 to number you want in the last 'fc_reg' layer(last fully connected layer) in both DiavNet_1.prototxt and DiavNet_Deploy_Res_1.prototxt.

If you want to train the network for your classification data instead, change your createLMDB file in same way as you'd do for single variable regression. In DiavNet_1.prototxt and DiavNet_Deploy_Res_1.prototxt, change the number of output neurons equal to your total number of class. Finally, in DiavNet_1.prototxt, change the euclidean loss to standard softmax loss and train.

##Some Practical Issues

Learning rate for classification and regression tasks are normally very different. For regression, if the network is diverging, start with a very small learning rate. Batch size makes a big impact on speed and converge, try to make it as large as you can.
