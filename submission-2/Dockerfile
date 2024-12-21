FROM tensorflow/serving:latest

COPY ./output/serving_model /models/imdb-review-model
COPY ./monitoring /monitoring

ENV MODEL_NAME=imdb-review-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/monitoring/prometheus.config"

ENTRYPOINT ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=imdb-review-model", "--model_base_path=/models/imdb-review-model", "--monitoring_config_file=/monitoring/prometheus.config"]