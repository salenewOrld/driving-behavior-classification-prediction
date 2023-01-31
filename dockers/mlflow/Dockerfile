FROM continuumio/miniconda3
RUN apt-get update
RUN apt install -y make
RUN conda install -c conda-forge mlflow 

EXPOSE 5959

#ENTRYPOINT [ "mlflow", 'server', '"--ip=0.0.0.0', '"--port=5959"' ]

#RUN mlflow server -h 0.0.0.0 -p 5959