FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

RUN apt-get update \
 && apt-get -y upgrade \
 && apt-get install -y git curl pip \
 && git clone https://github.com/VectorSpaceLab/OmniGen.git \
 && cd OmniGen \
 && pip install --break-system-packages -e . \
 && pip install --break-system-packages gradio spaces \
 && pip cache purge \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && sed -i 's/demo\.launch(.*)/demo.launch(server_name="0.0.0.0")/' app.py

WORKDIR /OmniGen/
ENTRYPOINT ["python3", "app.py"]
