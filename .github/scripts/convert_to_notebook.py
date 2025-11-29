"""
Convert Python example scripts to Jupyter notebooks for Colab integration.

This script is adapted from deepinv's conversion script and simplified for braindecode's needs.
It converts Sphinx-Gallery Python examples to Jupyter notebooks.
"""

import argparse
import copy
from pathlib import Path

import nbformat
from sphinx_gallery import gen_gallery
from sphinx_gallery.notebook import jupyter_notebook, save_notebook
from sphinx_gallery.py_source_parser import split_code_and_text_blocks


def convert_script_to_notebook(src_file: Path, output_file: Path, gallery_conf):
    """
    Convert a single Python script to a Jupyter notebook.
    
    Parameters
    ----------
    src_file : Path
        Path to the source Python script
    output_file : Path
        Path where the notebook should be saved
    gallery_conf : dict
        Sphinx-Gallery configuration dictionary
    """
    # Parse the Python file
    file_conf, blocks = split_code_and_text_blocks(str(src_file))

    # Convert to notebook (returns a dict, not a notebook object)
    example_nb_dict = jupyter_notebook(blocks, gallery_conf, str(src_file.parent))
    
    # Convert dict to nbformat notebook object
    example_nb = nbformat.from_dict(example_nb_dict)

    # Prepend an installation cell for braindecode
    # Check if first cell already has pip install
    try:
        first_source = example_nb.cells[0].source if example_nb.cells else ""
    except (IndexError, AttributeError):
        first_source = ""
    
    install_cmd = "%pip install braindecode"
    if "pip install" not in first_source or "braindecode" not in first_source:
        install_cell = nbformat.v4.new_code_cell(source=install_cmd)
        install_cell.metadata["language"] = "python"
        example_nb.cells.insert(0, install_cell)

    # Ensure the parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save notebook
    save_notebook(example_nb, output_file)
    print(f"Notebook saved to: {output_file}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Python example scripts to Jupyter notebooks."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the Python script to convert",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the converted notebook",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    target_path = Path(args.output)

    # Use default gallery configuration
    gallery_conf = copy.deepcopy(gen_gallery.DEFAULT_GALLERY_CONF)

    print(f"Processing: {input_path}")
    try:
        convert_script_to_notebook(input_path, target_path, gallery_conf)
        print(f"Successfully converted {input_path} to {target_path}")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        raise
