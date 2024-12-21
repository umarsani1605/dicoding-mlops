import os
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.components import init_components

PIPELINE_NAME = "umarsani16-pipeline"

# Pipeline inputs
DATA_ROOT = "/content/umarsani16-pipeline/data"
TRANSFORM_MODULE_FILE = "modules/transform_module_file.py"
TRAINER_MODULE_FILE = "modules/trainer_module_file.py"

# Pipeline outputs
OUTPUT_BASE = "output"
SERVING_MODEL_DIR = os.path.join(OUTPUT_BASE, 'serving_model')
PIPELINE_ROOT = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")

def init_local_pipeline(components, pipeline_root: Text) -> pipeline.Pipeline:
    """
    Inisialisasi pipeline lokal.

    Args:
        components: Komponen-komponen pipeline.
        pipeline_root (Text): Direktori root pipeline.

    Returns:
        pipeline.Pipeline: Objek pipeline yang telah diinisialisasi.
    """
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0"
    ]
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH),
    )

def main():
    """
    Fungsi utama untuk menjalankan pipeline.
    """
    logging.set_verbosity(logging.INFO)
    components = init_components(
        data_dir=DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR,
    )
    imdb_pipeline = init_local_pipeline(components, PIPELINE_ROOT)
    BeamDagRunner().run(pipeline=imdb_pipeline)

if __name__ == "__main__":
    main()
    