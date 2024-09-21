FROM quay.io/modh/cuda-notebooks@sha256:3beed917f90b12239d57cf49c864c6249236c8ffcafcc7eb06b0b55272ef5b55
RUN git clone https://github.com/eformat/langchain-chainlit-demo.git
WORKDIR /opt/app-root/src
RUN pip install --no-cache-dir -r langchain-chainlit-demo/requirements.txt
EXPOSE 8080
ENTRYPOINT chainlit run langchain-chainlit-demo/app.py -w --port 8080 --host 0.0.0.0
