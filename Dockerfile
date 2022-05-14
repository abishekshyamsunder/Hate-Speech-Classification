FROM ubuntu:20.04
RUN apt-get update \ 
    && apt-get install -y python3 python3-pip gcc g++ make git
RUN pip install jupyter
WORKDIR /workdir
RUN cd workdir; git clone https://github.com/abishekshyamsunder/Hate-Speech-Classification.git
RUN cd workdir; mkdir data_mount
RUN cd Hate-Speech-Classification; pip install -r requirements.txt
EXPOSE 8888