### Building the docker image command

```bash
    docker build -t mean-duration-prediction-service:v1 .
```
### Running the docker image command in interactive mode

```bash
    docker run -it --rm -p 9696:9696 mean-duration-prediction-service:v1
```