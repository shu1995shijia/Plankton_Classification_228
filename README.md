### Plankton Classification

The purpose of this project is to develop a plankton classification system based on deep learning
algorithms. Plankton are the diverse collection of organismsthat live in thewater . There are
hundreds of planktonic specieshave been discovered by scientists. Since the planktons are
diverse and their size is microscopic, it is inefficient to hand classify them. Machine learning for
the purpose of image classification has been heavily researched in recent years. Several deep
layer convolutional neural networks created by data scientists such as DenseNet have achieved
outstanding accuracy on image classification. We want to utilize these models to identify the
species of plankton.Training a neural network for this purpose is a challenging and interesting
application. We are looking forward to diving into this project and discovering its effectiveness
in a real application.

Please the run the jupyter notebook file in dataset folder.
   - data_preprocess.ipynb: The data preprocess file for organize the dataset
   - data_augmentation.ipynb: The file to perform data augmentation
   
For ResNet Model please refer to ResNet_Model folder.
   - plankton_dataloader.py: Dataloader used for the Pytorch model.
   - transfer_resnet.ipynb: For training the ResNet Model and storing the checkpoints, loss and accuracy
   - test_resnet.ipynb: For testing the model obtained from Training and perform evaluations.
   
For DenseNet Model please refer to DenseNet_Model folder.
   - plankton_dataloader.py: Dataloader used for the Pytorch model.
   - transfer_densenet.ipynb: For training the DenseNet Model and storing the checkpoints, loss and accuracy
   - test_densenet.ipynb: For testing the model obtained from Training and perform evaluations.
