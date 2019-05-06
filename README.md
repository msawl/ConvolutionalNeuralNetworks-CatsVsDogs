# ConvolutionalNeuralNetworks-CatsVsDogs
Creating a model to diffrentiate between 2 diffrent categories of images. In this case its Cats Vs Dogs. You can use any other category too.

## To create a CNN model:-
1. Use Conda to load all the dependencies using this command 
```conda env create -f environment.yml``` 
This will load all the dependencies that are needed
2. Now do ```conda activate cats_vs_dogs```
3. Now run the create_model.py file to generate the model to detect images using this command ```python create_model.py```
4. Now you will see a model file named "model.h5" and "model.json"

## To use the model generated above to detect images:-
1. Open the "test_model.py" file and look at line 25. You can edit the filename '4.jpg' to any other file.
2. Now run the file using ```python test_model.py```
3. You will see the interpretation that our model made
