# chainlit

Run locally (should work with python3.9+)

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Database

```bash
podman run -d --name postgres \
-e POSTGRESQL_USER=user \
-e POSTGRESQL_PASSWORD=password \
-e POSTGRESQL_ADMIN_PASSWORD=password \
-e POSTGRESQL_DATABASE=vectordb \
-p 5432:5432 \
quay.io/rh-aiservices-bu/postgresql-15-pgvector-c9s:latest

podman exec -it postgres psql -d vectordb -c "CREATE EXTENSION vector;"
```

ChatBot

```bash
chainlit run app.py -w
```

Build image in OpenShift

```bash
oc -n openshift new-build \
  --strategy docker --dockerfile - --name chatbot < Containerfile
```
