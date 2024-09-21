# chainlit

Run locally

```bash
chainlit run app.py -w
```

Build image in OpenShift

```bash
oc -n openshift new-build \
  --strategy docker --dockerfile - --name chatbot < Containerfile
```
