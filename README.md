# Hate-Speech-Classification
Hate speech is defined as abusive or threatening speech or writing that expresses prejudice against a particular group, especially on the basis of race, religion, disability, gender identity or sexual orientation, etc.  
This repository contains code to build different Neural network models for identifying which model provides highest accuracy while classifying sentences based on whether they contain hate speech or not.  
Primarily, the models will be trained on datasets obtained from different social media platforms using tensorflow as the framework.   

## Prerequisites  
For running the code in this repository 1 of 2 methods can be used.  
Method 1 can be used for running the code immediately however for persistent long term use Method 2 is recommended.  
### Method 1: Using Python installed in machine and running code  
If you already have python3 setup in you machine, you just have to run the script  
```
git clone https://github.com/abishekshyamsunder/Hate-Speech-Classification.git
cd Hate-Speech-Classification
make
```
We recommend running this code in a new virtual environment, as running `make` automatically installs the required packages for running the code in the repo. These might conflict with existing versions of those packages in your machine.  
To create and activate a new virtual environemnt please check [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi_7eHB8N_3AhV7g4kEHTjtBpIQFnoECA8QAQ&url=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fvenv.html&usg=AOvVaw1SQ6VGTcJCX7W6wOs1SpnV)  
### Method 2: Using dedicated docker image from Docker Hub and running the code  
For using this method, you must install the docker application in you local machine.  
Please follow the instructions [here](https://docs.docker.com/get-docker/) to setup docker in you machine.  
Once this step is completed, you can run the following script  
```
docker pull abishekshyamsunder/hate-speech-classifier-image
docker run -it -p 8888:8888 abishekshyamsunder/hate-speech-classifier-image  
cd Hate-Speech-Classification
make
```
This will create a container of the docker image, that contains the necessary environment to run the code.  

## Understanding the Makefile    
**The Makefile is the point of entry** for running the different python scripts in this repository.  
`pip install -r requirements.txt` installs all the python packages required.  
`python extract_and_clean.py` extracts all the data from their respective repositories, cleans them and stores them in the folder data/clean  
All the other python commands, train and test different models on this data (each one is explained in its own section).     

## Extracting and cleaning data  
**We aim to provide models that can be generalised across different social media platforms** thus two datasets have been used 
1. [Twitter Hate Speech Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)   
2. [Reddit Hate Speech Dataset](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech)  

Each of these require different cleaning strategies to bring them to the form Data, Label.  
These have been implemented in the `extract_and_clean.py` file.  
Throughout the training and testing pipeline, 2 versions of each model will exist, one trained on the combined dataset and the other just on the Reddit Dataset. This is because the Reddit Dataset in general has longer sentences and the statistics obtained for models trained on this dataset will have high contrast between models that can't handle long sentences (e.g. Simple Dense) and models that can handle long sentences (e.g. Bi-LSTM).  

## Pipeline Components for Preprocessing and training  
All the preprocessing, training and testing has been neatly organised into pipelines.  
The different components of the pipelines have been defined in the file `function_generators.py`.  
These components must be defined in classes containing the fit and transform method so that they can be integrated into the sklearn pipelines smoothly.  

## Candidate Models  
Initially, two simple ML models are trained: Naive Bayes Classifier and a Support Vector Classifier.  These are used to set the baseline performance in terms of accuracy for the classification task.  
This is done by running the `python -W ignore baseline_model.py` script.  
It should be noted that the following statistics:  
1. Model Name    
2. Accuracy  
3. F1-Score  
4. Training Time  
5. Time to classify one instance (named testing time)  
are all stored for all models (including the baseline models) for analysis later.  

All the models used are simple 3-4 layered models proving yet again the Neural Networks are really powerful tools that can be used to solve complex problems easily.  
For each of the models mentioned below, 3 variations are trained  
1. Model by itself (described in its section, and identified by its name)  
2. The Model with a dropout layer added  
3. The Model with a Batch Norm layer added  

Because these tensorflow models needed to be added to a scikit-learn pipeline, there were wrapped as a Keras Classifier object and could directly be added to the pipeline. Though this adds a level of complexity to the model, this helps maintain clean and clear code.  

### Simple Dense  
This is the simplest NN model used for training with just a 10 unit Dense layer between the embedding layer and the output layer.  
Upon training, this model provided an accuracy of approximately ~77%, that far surpassed the accuracies obtained by the baseline ML models.  

### Simple RNNs  
These **Recurrent Neural Networks**, specially designed for dealing with temporal data had a very large training time, but provided accuracies greater than the simple Dense Networks.  



