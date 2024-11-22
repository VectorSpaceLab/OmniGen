# Build the Docker Image
Run the following commands to build the image:

```bash
cd docker/
sudo docker build -t omni1 .
```

# Run the Docker Container
Use this command to run the container:

```bash
mkdir -p ~/Omni/.cache
sudo docker run -p 7860:7860 --name omni --rm --gpus all -it -v ~/Omni/.cache:/root/.cache omni1
```

**Model Storage**: The `/root/.cache` directory in the container is where the model will be saved. Ensure you have at least 18GB of free space.

**First Run**: The first time you run the container, it may take several minutes to download the model.

**Subsequent Runs**: Starting the container will take about 10–30 seconds.
