# llm_context_os/utils/hf_downloader.py
import typing as t
from pathlib import Path
import shutil
import os # For environment variables, though hf_hub handles it mostly

# Hugging Face Hub imports
HUGGINGFACE_HUB_AVAILABLE = False
hf_hub_download = None
snapshot_download = None
EntryNotFoundError = None
HfHubHTTPError = None # For specific HTTP errors
RepositoryNotFoundError = None # More specific than EntryNotFoundError for repos
RevisionNotFoundError = None # For invalid revisions

try:
    from huggingface_hub import (
        hf_hub_download as _hf_hub_download,
        snapshot_download as _snapshot_download,
        EntryNotFoundError as _EntryNotFoundError, # For files mostly
        HfHubHTTPError as _HfHubHTTPError,
        RepositoryNotFoundError as _RepositoryNotFoundError, # For missing repos
        RevisionNotFoundError as _RevisionNotFoundError # For bad revisions
    )
    hf_hub_download = _hf_hub_download
    snapshot_download = _snapshot_download
    EntryNotFoundError = _EntryNotFoundError
    HfHubHTTPError = _HfHubHTTPError
    RepositoryNotFoundError = _RepositoryNotFoundError
    RevisionNotFoundError = _RevisionNotFoundError
    HUGGINGFACE_HUB_AVAILABLE = True
    print("Successfully imported huggingface_hub components for hf_downloader.")
except ImportError:
    print("Warning: huggingface_hub library not found. Model downloading from Hugging Face Hub will not function.")


def download_model_from_hf(
    repo_id: str,
    target_dir: Path,
    filename: t.Optional[str] = None,
    hf_token: t.Optional[str] = None, # If None, huggingface_hub will try to use env var or cached token
    ignore_patterns: t.Optional[t.List[str]] = None,
    allow_patterns: t.Optional[t.List[str]] = None,
    repo_type: t.Optional[str] = None, # e.g., "model", "dataset", "space"
    revision: t.Optional[str] = None  # e.g., "main", "v1.0", commit hash
) -> t.Tuple[bool, t.Union[str, Path]]:
    """
    Downloads a model file or an entire repository snapshot from the Hugging Face Hub.

    Args:
        repo_id: The ID of the repository (e.g., "google-bert/bert-base-cased").
        target_dir: The base directory where the downloaded content should be saved.
                    A subdirectory named after the repo_id's last part will be created here.
        filename: If specified, downloads only this single file from the repo.
                  Otherwise, downloads the entire repository snapshot.
        hf_token: Optional Hugging Face API token for private repositories.
        ignore_patterns: For snapshot download, list of glob patterns to ignore.
        allow_patterns: For snapshot download, list of glob patterns to allow (takes precedence).
        repo_type: For snapshot download, type of the repo.
        revision: Specific model revision (branch, tag, commit hash).

    Returns:
        A tuple (success: bool, result: Union[str, Path]).
        If success is True, result is the Path to the downloaded file or directory.
        If success is False, result is an error message string.
    """
    if not HUGGINGFACE_HUB_AVAILABLE or not hf_hub_download or not snapshot_download or \
       not EntryNotFoundError or not HfHubHTTPError or not RepositoryNotFoundError or not RevisionNotFoundError:
        msg = "huggingface_hub library or specific exceptions are not available. Cannot download models."
        print(f"[HF Downloader] {msg}")
        return False, msg

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[HF Downloader] Base target directory ensured: {target_dir.resolve()}")

        # Determine the specific subdirectory for this repo
        repo_name_part = repo_id.split('/')[-1]
        model_or_repo_specific_target_dir = target_dir / repo_name_part

        if filename:
            # --- Single File Download ---
            model_or_repo_specific_target_dir.mkdir(parents=True, exist_ok=True)
            final_file_path = model_or_repo_specific_target_dir / filename

            print(f"[HF Downloader] Starting download of single file: {filename} from repo: {repo_id} (revision: {revision or 'main'}) to '{final_file_path}'")

            cached_file_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=hf_token,
                repo_type=repo_type,
                revision=revision
            )

            final_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached_file_path_str, final_file_path)

            print(f"[HF Downloader] File {filename} successfully downloaded and copied to: {final_file_path}")
            return True, final_file_path

        else:
            # --- Repository Snapshot Download ---
            print(f"[HF Downloader] Starting download of entire repo snapshot: {repo_id} (revision: {revision or 'main'}) to '{model_or_repo_specific_target_dir}'")

            snapshot_path_str = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=hf_token,
                ignore_patterns=ignore_patterns,
                allow_patterns=allow_patterns,
                local_dir=str(model_or_repo_specific_target_dir),
                local_dir_use_symlinks=False
            )

            print(f"[HF Downloader] Repo {repo_id} successfully downloaded to: {snapshot_path_str}")
            return True, Path(snapshot_path_str)

    except EntryNotFoundError:
        msg = f"File '{filename}' or repo '{repo_id}' (revision: {revision or 'main'}) not found or not accessible with provided token."
        print(f"[HF Downloader] {msg}")
        return False, msg
    except RepositoryNotFoundError:
        msg = f"Repository not found: '{repo_id}'. Please check the repository ID and network."
        print(f"[HF Downloader] {msg}")
        return False, msg
    except RevisionNotFoundError:
        msg = f"Revision '{revision}' not found for repo '{repo_id}'."
        print(f"[HF Downloader] {msg}")
        return False, msg
    except HfHubHTTPError as e:
        msg = f"HTTP error accessing repo '{repo_id}' (file: {filename or 'snapshot'}, revision: {revision or 'main'}): {e}. Check network/token."
        print(f"[HF Downloader] {msg}")
        return False, msg
    except Exception as e:
        msg = f"An unexpected error occurred for '{repo_id}' (file: {filename or 'snapshot'}, revision: {revision or 'main'}): {type(e).__name__} - {str(e)}"
        print(f"[HF Downloader] {msg}")
        return False, msg
