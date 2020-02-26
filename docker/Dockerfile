# Select image to use as base for your specific model
FROM pytorch/pytorch

# Docker commands to run the container
LABEL build="docker build -t cardamageimage3 -f docker/Dockerfile ."
LABEL run="docker run --name=cardamage -d -p 7011:7000 cardamageimage3"

# Install git, dvc and dvc azure
RUN apt-get install git
#RUN pip install dvc
#RUN pip install dvc[azure]

# Download repository
#RUN git clone https://8d9761dfe571b84463a866dd5faadc5166f5da95@github.com/jhairgallardo/car-damage-recognition-demo.git

# Copy config file 
RUN mkdir -p /workspace/car-damage-recognition-demo/src
#COPY . /workspace/car-damage-recognition-demo/
COPY . /workspace/car-damage-recognition-demo/src/

# Set the working directory to repo
WORKDIR /workspace/car-damage-recognition-demo/src

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port available to the world outside this container
EXPOSE 7000 

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]