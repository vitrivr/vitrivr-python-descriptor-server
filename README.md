# vitrivr-python-descriptor-server

## Dependencies 
Software    | Version |
------------|---------|
Python      | 3.9     |

## In docker
```bash
git clone git@github.com:vitrivr/vitrivr-python-descriptor-server.git
cd ./vitrivr-python-descriptor-server
```

The following command builds and runs the docker container. Further it installs all python dependencies.
```bash
sudo docker run -p 3000:3000 $(sudo docker build -q .)
```
> [!NOTE]
>  On first start the models will be downloaded and output is quite for some time
