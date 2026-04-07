import json
from pathlib import Path

PARAMS_FILENAME = "params.safetensors"
HISTORY_FILENAME = "history.json"
METADATA_FILENAME = "metadata.json"
README_FILENAME = "README.md"


def tutorial_repo_id(tutorial_name: str, namespace: str = "braindecode") -> str:
    """Return the model repo id used for a tutorial checkpoint."""
    return f"{namespace}/{Path(tutorial_name).stem}"


def load_tutorial_artifact_paths(
    repo_id: str,
    *,
    filenames: tuple[str, ...],
    revision: str = "main",
):
    """Download arbitrary tutorial artifact files from the Hub and return local paths."""
    return _load_tutorial_hub_files(
        repo_id,
        filenames=filenames,
        revision=revision,
    )


def load_tutorial_metadata(repo_id: str, *, revision: str = "main") -> dict | None:
    """Load tutorial metadata from the Hub if available."""
    artifacts = _load_tutorial_hub_files(
        repo_id,
        filenames=(METADATA_FILENAME,),
        revision=revision,
    )
    if artifacts is None:
        return None

    try:
        return json.loads(Path(artifacts[METADATA_FILENAME]).read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_tutorial_hub_files(
    repo_id: str,
    *,
    filenames: tuple[str, ...],
    revision: str = "main",
):
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return None

    try:
        return {
            filename: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
            for filename in filenames
        }
    except (
        EntryNotFoundError,
        FileNotFoundError,
        HfHubHTTPError,
        OSError,
        RepositoryNotFoundError,
    ):
        return None


def load_tutorial_checkpoint_metadata(
    clf,
    repo_id: str,
    *,
    revision: str = "main",
    use_safetensors: bool = True,
) -> dict | None:
    """Load classifier parameters, history, and optional metadata from the Hub."""
    params_name = PARAMS_FILENAME if use_safetensors else "params.pt"
    artifacts = _load_tutorial_hub_files(
        repo_id,
        filenames=(params_name, HISTORY_FILENAME, METADATA_FILENAME),
        revision=revision,
    )
    if artifacts is None:
        return None

    metadata: dict = {}
    if METADATA_FILENAME in artifacts:
        try:
            metadata = json.loads(Path(artifacts[METADATA_FILENAME]).read_text())
        except (json.JSONDecodeError, OSError):
            pass

    clf.initialize()
    clf.load_params(
        f_params=artifacts[params_name],
        f_history=artifacts[HISTORY_FILENAME],
        use_safetensors=use_safetensors,
    )
    return metadata


def save_tutorial_checkpoint(
    clf,
    output_dir: str | Path,
    *,
    metadata: dict | None = None,
    readme_text: str | None = None,
    use_safetensors: bool = True,
) -> Path:
    """Save classifier parameters, history, and optional metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / (PARAMS_FILENAME if use_safetensors else "params.pt")
    clf.save_params(
        f_params=params_path,
        f_history=output_dir / HISTORY_FILENAME,
        use_safetensors=use_safetensors,
    )

    if metadata is not None:
        (output_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    if readme_text is not None:
        (output_dir / README_FILENAME).write_text(readme_text)

    return output_dir


def upload_tutorial_artifacts(
    repo_id: str,
    artifact_dir: str | Path,
    *,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
) -> str:
    """Upload a saved tutorial artifact directory to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to upload tutorial artifacts."
        ) from exc

    token = token or get_token()
    if token is None:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or log in with huggingface_hub."
        )

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(artifact_dir),
        commit_message=commit_message or f"Update tutorial artifacts for {repo_id}",
    )
    return f"https://huggingface.co/{repo_id}"
