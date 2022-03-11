# <div align='center'>Perform an hyperparameter optimization for SPANet</div>

## Create a docker image with the optimizer script (only if image doesn't already exists)

If ```spanet_hpo.py``` was modified, create a new docker image (otherwise, use gitlab-registry.cern.ch/jbossios/docker-images/atlas-spanet-jona-v17).

Run the following to create a new docker image (change CUSTOM accordingly):

sudo docker login gitlab-registry.cern.ch
sudo docker build . -f Dockerfile -t gitlab-registry.cern.ch/jbossios/docker-images/CUSTOM
sudo docker push gitlab-registry.cern.ch/jbossios/docker-images/CUSTOM

Repository with docker images: https://gitlab.cern.ch/jbossios/docker-images

## Submit job with Katib

1. Go to https://ml.cern.ch
2. Open the Katib tab
3. Click on ```Hyperparameter Tuning```
4. Copy the content of the yaml file (example: spanet_hpo_cpu_v17_11032022.yaml) and click on ```DEPLOY```
5. Job can be monitored under Katib > HP > Monitor

## TODO: add instructions on how to find optimal choice using all the outputs
