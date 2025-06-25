# Build the Docker Image
Run the following commands to build the image:

```bash
cd docker/
sudo docker build -t omni1 .
```

**Build Notes**:
- On a machine with a Xeon Silver CPU and an SSD, the build process takes approximately 10 minutes.
- The resulting Docker image is approximately 11GB in size.

# Run the Docker Container
Use this command to run the container:

```bash
mkdir -p ~/Omni/.cache
sudo docker run -p 7860:7860 --name omni --rm --gpus all -it -v ~/Omni/.cache:/root/.cache omni1
```

**Model Storage**: The directory `/root/.cache` inside the container is used to store the model. Make sure your system has at least 18GB of free space in the `~/Omni/.cache` directory.

**First Run**: The first time you run the container, it may take several minutes to download the model.

**Subsequent Runs**: Starting the container will take about 15–40 seconds.
