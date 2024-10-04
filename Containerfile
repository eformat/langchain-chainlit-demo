FROM quay.io/modh/cuda-notebooks@sha256:d8295bcf45a6a522f78c07dc62634a7775fc434621973bd81db243a5a63a1ffa
WORKDIR /opt/app-root/src
RUN git clone https://github.com/eformat/langchain-chainlit-demo.git
RUN pip install --no-cache-dir -r langchain-chainlit-demo/requirements.txt
RUN mkdir -p .chainlit && chmod 775 .chainlit
RUN cp langchain-chainlit-demo/.chainlit/config.toml .chainlit/config.toml
RUN cp -R langchain-chainlit-demo/public .
EXPOSE 8080
ENTRYPOINT chainlit run langchain-chainlit-demo/app.py -w --port 8080 --host 0.0.0.0 --headless
