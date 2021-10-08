FROM gitlab-registry.cern.ch/ai-ml/kubeflow_images/pytorch-notebook-gpu-1.8.1:v0.6.1-python3.8-atlas-v4

COPY spanet_hpo.py /spanet_hpo.py
