"""
Hugging Face Hub integration: download pretrained weights from and upload trained
weights to model repositories. Weights are stored in the self-contained npz format of
:mod:`cod_vae.checkpoint`, so downloads only require numpy. Requires the
huggingface_hub package (``pip install cod-vae[hub]``).
"""

from __future__ import annotations

from pathlib import Path

from .checkpoint import Params, load_npz, save_npz
from .config import CODVAEConfig

__all__ = ["DEFAULT_WEIGHTS_FILENAME", "download_pretrained", "push_to_hub"]

DEFAULT_WEIGHTS_FILENAME = "model.npz"


def _hf_hub():
    try:
        import huggingface_hub
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for Hugging Face Hub support. "
            "Install it with: pip install cod-vae[hub]"
        ) from e
    return huggingface_hub


def download_pretrained(
    repo_id: str,
    filename: str = DEFAULT_WEIGHTS_FILENAME,
    revision: str | None = None,
) -> tuple[CODVAEConfig, Params]:
    """Download and load a weights file from a Hugging Face Hub model repository."""
    path = _hf_hub().hf_hub_download(repo_id, filename, revision=revision)
    return load_npz(path)


def push_to_hub(
    repo_id: str,
    config: CODVAEConfig,
    params: Params,
    filename: str = DEFAULT_WEIGHTS_FILENAME,
    private: bool = False,
    commit_message: str = "Upload COD-VAE weights",
) -> str:
    """
    Upload weights to a Hugging Face Hub model repository (created if it does not
    exist). Returns the repository URL.
    """
    import tempfile

    hf = _hf_hub()
    api = hf.HfApi()
    repo_url = api.create_repo(repo_id, private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / Path(filename).name
        save_npz(path, config, params)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )
    return str(repo_url)
