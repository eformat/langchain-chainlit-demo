# 2025.1 - oc get is minimal-gpu -n redhat-ods-applications -o yaml
FROM quay.io/modh/odh-workbench-jupyter-minimal-cuda-py311-ubi9@sha256:7b335c289a48e71a6fc149ead252b5a2dafeae730a5108adf9ba7ee12b181ca2
WORKDIR /opt/app-root/src
RUN git clone https://github.com/eformat/langchain-chainlit-demo.git
RUN pip install --no-cache-dir -r langchain-chainlit-demo/requirements.txt
RUN mkdir -p .chainlit && chmod 775 .chainlit
RUN cp langchain-chainlit-demo/.chainlit/config.toml .chainlit/config.toml
RUN cp -R langchain-chainlit-demo/public .
EXPOSE 8080
ENTRYPOINT chainlit run langchain-chainlit-demo/app.py -w --port 8080 --host 0.0.0.0 --headless
