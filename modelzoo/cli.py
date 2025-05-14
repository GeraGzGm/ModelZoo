import os
import click

@click.command()
@click.option("--run_type", type=click.Choice(["train", "inference"], case_sensitive=False), required=True)
@click.option("--config_file", type=click.Path(exists=True), required=True)
@click.option("--model_path", type=click.Path(exists=True), required=False)
@click.option("--out_dir", type=click.Path(file_okay=False, dir_okay=True, writable=True), required=False)
def main(run_type, config_file, model_path, out_dir):
    if os.getenv("_MODELZOO_COMPLETE"):
        return  # Skip execution during completion
    from .main import real_main
    real_main(run_type, config_file, model_path, out_dir)