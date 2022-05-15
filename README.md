# Hate-Speech-Classification
Hate speech is defined as abusive or threatening speech or writing that expresses prejudice against a particular group, especially on the basis of race, religion, disability, gender identity or sexual orientation, etc.  
This repository contains code to build different Neural network models for identifying which model provides highest accuracy while classifying sentences based on whether they contain hate speech or not.  
Primarily, the models will be trained on datasets obtained from different social media platforms using tensorflow as the framework.   
The overall structure of the repo is as given below:  
<img width="431" alt="Screenshot 2022-05-15 at 1 53 02 PM" src="https://user-images.githubusercontent.com/49049340/168486922-0eb5cef9-2cef-4581-b6c9-d15ea3b8ac03.png">  


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
We have also provided the docker file that was used to create this image for reference.  

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
The code for training Simple Dense models can be run by `python -W ignore simple_model.py`  

### Simple RNNs  
These **Recurrent Neural Networks**, specially designed for dealing with temporal data had a very large training time, but provided accuracies greater than the simple Dense Networks.  
Again, the code for training the Simple RNN models can be run by `python -W ignore simple_rnn.py`  

### LSTMs and Bi-LSTMs  
LSTMs or Long Short term memory cells were specifically developed to tackle the problem of RNNs forgetting context. To understand this, something referenced in the current sentence could have been introduced 2 sentences ago and could be important in classifying sentences.  
Bi-Directional LSTMs are an upgrade over LSTMs capable of parsing the sentence in both the forward the reverse directions.  
The code for training LSTMs and Bi-LSTMs can be run using the commands  
- `python -W ignore lstm_model.py`  
- `python -W ignore bi_lstm_model.py`  

### GRUs and Bi-Directional GRUs  
Gated Recurrent Units are cells that are very similar to LSTMs in architecture. They contain 2 gates as opposed to 3 in LSTM.  
GRUs are meant to be more efficient than LSTMs and expected to perform better when there is only limited data while training.  
The GRU and Bi-Directional GRU models can be trained and tested using the commands  
- `python -W ignore gru_model.py`  
- `python -W ignore bi_gru_model.py`  

### Convolutional Layers  
Even through Convolution layers are primarily used in Image processing tasks, 1D Conv layers were used in the models here (both standalone and stacked) and their performance was evaluated.  
These models can be trained and tested using the command `python -W ignore conv_model.py`   

### Transfer Learning  
2 Methods of transfer learning were candidates for this task.  
#### Transfer Learning with GloVe embeddings  
The 300 dimension GloVe embeddings trained on the 6B corpus was obtained from the stanfor repository, from which only the embeddings belonging to words in our vocabulary were filtered out and this was used to replace the weights in the embedding layer.   
A GRU layer was used in this model.  
This code can be run using the command `python -W ignore glove_model.py`  


#### Transfer Learning with Bert   


## Batch Training and testing   
Along with the docker environment, two files named checking_test.py and train_saved.py have been provided that can be used for training a saved model on new data, or generating predictions for new data. These can be run by binding the new data directory to the docker container and calling the appropriate commands  
```
docker run -it -p 8888:8888 --mount type=bind,source=<absolute path to data folder>,target=/workdir/data_mount abishekshyamsunder/hate-speech-classifier-image
cd Hate-Speech-Classification; python3 checking_test.py /workdir/data_mount/clean/new_data.csv /workdir/data_mount/preds.txt
cd Hate-Speech-Classification; python3 train_saved.py /workdir/data_mount/clean/new_data.csv
```
The models can be saved by using a callback, which is provided in the `simple_model.py` file (and can be used in any other training file). 
The advantage of the above method is that we are binding a folder in our container with a local folder, thus any changes made in the container will be persistent and will be reflected in our local directory as well.  


## Analysis  
A final analysis of the stats is present in the `analysis.ipynb` file (that can be viewed directly from GitHub).  
It tells us which models have the best accuracy and which ones take less time to train, leaving us with the correct information for choosing a trade-off.  


## Results  
Given below are the results observed post training, during the analysis phase.  

![Combined Time Vs Acc](https://user-images.githubusercontent.com/49049340/168451960-b371ad22-c2f7-4005-b3ee-93ad13952e55.png)
![Reddit Time Vs Acc](https://user-images.githubusercontent.com/49049340/168451968-72fa5611-4398-4c99-b01f-b56549c3d673.png)

- Max Accuracy in Reddit is Higher than Max Accuracy in the Combined Dataset
- All Neural Network Models perform much better than simple ML classification models  
- Highest accuracy with less time is given by Bi-LSTM with Batch Norm  
- RNNs take too long to train, and give bad performance  
- Note that in both cases, CONV stacked performs better than one with a single CONV layer as it has a bigger receptive field, and is thus able to handle longer sentences. (77% and 76% acc respectively)


<img width="417" alt="Screenshot 2022-05-14 at 3 22 02 PM" src="https://user-images.githubusercontent.com/49049340/168452028-88117d12-1265-4f04-aef6-970be815b475.png">
<img width="417" alt="Screenshot 2022-05-14 at 3 22 12 PM" src="https://user-images.githubusercontent.com/49049340/168452034-1e629fef-18d2-4ae8-8749-62071db8ef72.png">

- In both cases, transfer learning with GloVe was not beneficial. 16 dimensional embeddings was more than sufficient to work with the data.  

<img width="417" alt="Screenshot 2022-05-14 at 3 17 58 PM" src="https://user-images.githubusercontent.com/49049340/168452039-2f201fce-951f-48e6-b506-08d679692024.png">
<img width="417" alt="Screenshot 2022-05-14 at 3 19 12 PM" src="https://user-images.githubusercontent.com/49049340/168452041-19de437b-ea69-4e26-b225-ae86a868608f.png">

- Bi-LSTM with Batch norm always seems to outperform all other architectures.  

<img width="431" alt="Screenshot 2022-05-14 at 3 30 21 PM" src="https://user-images.githubusercontent.com/49049340/168452045-cc6e5d4e-dede-4934-8f11-e2bfa09a4e58.png">
**We know that LSTM with Batch Norm performs the best, but we can also see that the time it takes to process a sentence is almost double that of Simple NN. This might pose as a problem when if need to deploy our model to serve millions of requests per second (as in a real time on the fly hate speech classifier)   **





