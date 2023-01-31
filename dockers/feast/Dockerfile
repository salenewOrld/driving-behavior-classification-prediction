FROM openjdk:8

RUN apt-get update && apt-get install -y git ant

WORKDIR /usr/src

ADD . ./

#ENTRYPOINT [ "/opt/app/run.sh", "--port", "8080" ]