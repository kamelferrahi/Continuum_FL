# CONTINUUM-FEDHE-Graph 

Welcome to the official repository housing the FEDHE-Graph implementation for our solution Continuum! This repository provides you with the necessary tools and resources to leverage federated learning techniques within the context of Continuum, a comprehensive framework for federated learning research.

![architecture(1)-1](https://github.com/kamelferrahi/MAGIC_FEDERATED_FedML/assets/72205931/f3e67d1f-2fa1-4800-81e6-7d9c5e509cf7)

Original project: https://github.com/FDUDSDE/MAGIC

## Environment Setup

The command are used in an environnement that consist of Ubuntu 22.04 with miniconda installed

Original project: https://github.com/FDUDSDE/MAGIC

First create the conda environnement for fedml with MPI support 

```
conda create --name fedml-pip python=3.8
conda activate fedml-pip
conda install --name fedml-pip pip
conda install -c conda-forge mpi4py openmpi
pip install "fedml[MPI]" 
```

Clone the MAGIC FedML project onto your current folder 

```
git clone https://github.com/kamelferrahi/MAGIC_FEDERATED_FedML
```

Install the necessary packages for Magic to run

```
conda install -c conda-forge aiohttp=3.9.1 aiosignal=1.3.1 anyio=4.2.0 attrdict=2.0.1 attrs=23.2.0 blis=0.7.11 boto3=1.34.12 botocore=1.34.12 brotli=1.1.0 catalogue=2.0.10 certifi=2023.11.17 chardet=5.2.0 charset-normalizer=3.3.2 click=8.1.7 cloudpathlib=0.16.0 confection=0.1.4 contourpy=1.2.0 cycler=0.12.1 cymem=2.0.8 dgl=1.1.3 dill=0.3.7 fastapi=0.92.0 fedml=0.8.13.post2 filelock=3.13.1 fonttools=4.47.0 frozenlist=1.4.1 fsspec=2023.12.2 gensim=4.3.2 gevent=23.9.1 geventhttpclient=2.0.9 gitdb=4.0.11 GitPython=3.1.40 GPUtil=1.4.0 graphviz=0.8.4 greenlet=3.0.3 h11=0.14.0 h5py=3.10.0 httpcore=1.0.2 httpx=0.26.0 idna=3.6 Jinja2=3.1.2 jmespath=1.0.1 joblib=1.3.2 kiwisolver=1.4.5 langcodes=3.3.0 MarkupSafe=2.1.3 matplotlib=3.8.2 mpi4py=3.1.3 mpmath=1.3.0 multidict=6.0.4 multiprocess=0.70.15 murmurhash=1.0.10 networkx=2.8.8 ntplib=0.4.0 numpy=1.26.3 nvidia-cublas-cu12=12.1.3.1 nvidia-cuda-cupti-cu12=12.1.105 nvidia-cuda-nvrtc-cu12=12.1.105 nvidia-cuda-runtime-cu12=12.1.105 nvidia-cudnn-cu12=8.9.2.26 nvidia-cufft-cu12=11.0.2.54 nvidia-curand-cu12=10.3.2.106 nvidia-cusolver-cu12=11.4.5.107 nvidia-cusparse-cu12=12.1.0.106 nvidia-nccl-cu12=2.18.1 nvidia-nvjitlink-cu12=12.3.101 nvidia-nvtx-cu12=12.1.105 onnx=1.15.0 packaging=23.2 paho-mqtt=1.6.1 pandas=2.1.4 pathtools=0.1.2 pillow=10.2.0 preshed=3.0.9 prettytable=3.9.0 promise=2.3 protobuf=3.20.3 psutil=5.9.7 py-machineid=0.4.6 pydantic=1.10.13 pyparsing=3.1.1 python-dateutil=2.8.2 python-rapidjson=1.14 pytz=2023.3.post1 PyYAML=6.0.1 redis=5.0.1 requests=2.31.0 s3transfer=0.10.0 scikit-learn=1.3.2 scipy=1.11.4 sentry-sdk=1.39.1 setproctitle=1.3.3 shortuuid=1.0.11 six=1.16.0 smart-open=6.3.0 smmap=5.0.1 sniffio=1.3.0 spacy=3.7.2 spacy-legacy=3.0.12 spacy-loggers=1.0.5 SQLAlchemy=2.0.25 srsly=2.4.8 starlette=0.25.0 sympy=1.12 thinc=8.2.2 threadpoolctl=3.2.0 torch=2.1.2 torch-cluster=1.6.3 torch-scatter=2.1.2 torch-sparse=0.6.18 torch-spline-conv=1.2.2 torch_geometric=2.4.0 torchvision=0.16.2 tqdm=4.66.1 triton=2.1.0 tritonclient=2.41.0 typer=0.9.0 typing_extensions=4.9.0 tzdata=2023.4 tzlocal=5.2 urllib3=2.0.7 uvicorn=0.25.0 wandb=0.13.2 wasabi=1.1.2 wcwidth=0.2.12 weasel=0.3.4 websocket-client=1.7.0 wget=3.2 yarl=1.9.4 zope.event=5.0 zope.interface=6.1
```

Finally run the federated algorithm using the mpi command 

```
hostname > mpi_host_file
mpirun -np 4  -hostfile mpi_host_file --oversubscribe python main.py --cf fedml_config.yaml
```

## Federated learning parameters
You can adjust federated learning parameters in the `fedml_config.yaml` file. 
Parameters such as aggregation algorithm, number of clients, and clients per round for aggregation can be modified:
```
train_args:
  federated_optimizer: "FedAvg"
  client_id_list: 
  client_num_in_total: 4
  client_num_per_round: 4
```

The algorithm tested are `FedAvg`, `FedProx` and `FedOpt`

## Datasets
The experiments utilize datasets similar to those in the original Magic project. To change datasets, edit the `fedml_config.yaml` file:
  ```
data_args:
  dataset: "wget"
  data_cache_dir: ~/fedgraphnn_data/
  part_file:  ~/fedgraphnn_data/partition
```

Feel free to explore and modify these settings according to your specific requirements!



