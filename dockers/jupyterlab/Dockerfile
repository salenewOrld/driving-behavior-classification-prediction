FROM continuumio/anaconda3
RUN apt-get update
RUN apt install -y make
#RUN conda install -c conda-forge gcc_linux-64
RUN conda config --add channels conda-forge
COPY conda-requirements.txt .
RUN conda install -y --file conda-requirements.txt
COPY pip_requirements.txt .
RUN pip3 install -r pip_requirements.txt

EXPOSE 8888
ENTRYPOINT [ "jupyter", "lab", "--ip='*'","--port=8888", "--no-browser","--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root", "--autoreload"]
