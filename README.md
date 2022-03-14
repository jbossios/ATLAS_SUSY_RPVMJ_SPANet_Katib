# <div align='center'>Perform an hyperparameter optimization for SPANet</div>

## Create a docker image with the optimizer script (only if image doesn't already exists)

If ```spanet_hpo.py``` was modified, create a new docker image (otherwise, use gitlab-registry.cern.ch/jbossios/docker-images/atlas-spanet-jona-v18).

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

## How to find optimal choice (set of hyperparameters that maximizes the reconstruction efficiency) using all the outputs

1. Get predictions from each network

First, open a jupyter notebook on Kubeflow following these steps:

- ssh -D 8090 lxplus.cern.ch
- google-chrome --proxy-server=socks5://127.0.0.1:8090
- Go to https://ml.cern.ch
- Create a notebook using 1 GPU and the following image: gitlab-registry.cern.ch/ai-ml/kubeflow_images/atlas-pytorch-gpu:0183442cdb7ad58434d6626b2ac6ff2befffa9a9

Second, set the following in the ```predict_hpo.py``` script and run it:

- ```PATH```: path to katib outputs (outputs should be within a folder inside this path)
- ```Version```: name of such a folder
- ```outPATH```: path where output H5 files will be sabed
- ```TestFile```: input signal H5 file used to predict assigments

2. Evaluate reconstruction efficiency for each network and find the most optimal network

Set the following in ```EvaluatePerformance_signal.py``` and run it:

- ```Tag```: should match with ```Version``` above
- ```Samples['True']```: should match with ```TestFile``` above.
