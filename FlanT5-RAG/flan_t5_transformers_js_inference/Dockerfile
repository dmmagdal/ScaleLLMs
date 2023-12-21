# Docker file to run a container that will run index.ts.

# Load NodeJS image for node 18.12.1.
# FROM node:18.12.1-slim
FROM node:18.19.0-slim

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /flan-t5-infernce-js
WORKDIR /flan-t5-infernce-js
ADD . /flan-t5-infernce-js

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN npm install --platform=linux --arch=arm64v8
# RUN npm run build

# Run index.js.
CMD ["node", "index.js"]

# docker build -t myapp . -> Build.
# docker run myapp -> Run the application.