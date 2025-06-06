import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import shutil
import os

# Adjust import path based on actual project structure
# Assuming llm_context_os is in the Python path or tests are run from project root
from llm_context_os.utils.hf_downloader import (
    download_model_from_hf,
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError
)

# Ensure HUGGINGFACE_HUB_AVAILABLE is True for tests, or mock its check
# For simplicity, we assume it's True if the module could be imported.
# If there's an explicit check for HUGGINGFACE_HUB_AVAILABLE in download_model_from_hf,
# it might need to be patched for these tests to run without actual hf_hub.

class TestHfDownloader(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("temp_downloader_tests_unit")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        # Create a dummy file to be "returned" by hf_hub_download mock
        self.dummy_cache_file = self.test_dir / "dummy_cached_file.gguf"
        with open(self.dummy_cache_file, "w") as f:
            f.write("dummy content")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch('llm_context_os.utils.hf_downloader.shutil.copy')
    @patch('llm_context_os.utils.hf_downloader.hf_hub_download')
    def test_download_single_file_success(self, mock_hf_download, mock_shutil_copy):
        mock_hf_download.return_value = str(self.dummy_cache_file)
        repo_id = "test/model-single"
        filename = "file.gguf"

        success, result_path = download_model_from_hf(
            repo_id=repo_id,
            target_dir=self.test_dir,
            filename=filename
        )

        self.assertTrue(success)
        expected_path = self.test_dir / repo_id.split('/')[-1] / filename
        self.assertEqual(result_path, expected_path)
        mock_hf_download.assert_called_once_with(
            repo_id=repo_id,
            filename=filename,
            token=None,
            repo_type=None,
            revision=None
        )
        mock_shutil_copy.assert_called_once_with(str(self.dummy_cache_file), expected_path)

    @patch('llm_context_os.utils.hf_downloader.snapshot_download')
    def test_download_snapshot_success(self, mock_snapshot_download):
        repo_id = "test/model-snapshot"
        expected_snapshot_path_str = str(self.test_dir / repo_id.split('/')[-1])
        mock_snapshot_download.return_value = expected_snapshot_path_str

        success, result_path = download_model_from_hf(
            repo_id=repo_id,
            target_dir=self.test_dir,
            # No filename implies snapshot
            ignore_patterns=["*.txt"],
            allow_patterns=["*.bin"],
            repo_type="model",
            revision="test-rev"
        )

        self.assertTrue(success)
        self.assertEqual(result_path, Path(expected_snapshot_path_str))
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            token=None,
            repo_type="model",
            revision="test-rev",
            ignore_patterns=["*.txt"],
            allow_patterns=["*.bin"],
            local_dir=expected_snapshot_path_str,
            local_dir_use_symlinks=False # Assuming this is the default in the actual code
        )

    @patch('llm_context_os.utils.hf_downloader.hf_hub_download', side_effect=EntryNotFoundError("File not found"))
    def test_download_file_not_found(self, mock_hf_download):
        success, message = download_model_from_hf(
            repo_id="test/nonexistent",
            target_dir=self.test_dir,
            filename="nonexistent.gguf"
        )
        self.assertFalse(success)
        self.assertIn("not found in repo", message.lower())
        self.assertIn("nonexistent.gguf", message)

    @patch('llm_context_os.utils.hf_downloader.snapshot_download', side_effect=RepositoryNotFoundError("Repo not found"))
    def test_download_repo_not_found(self, mock_snapshot_download):
        success, message = download_model_from_hf(
            repo_id="unknown/repo",
            target_dir=self.test_dir
            # No filename implies snapshot
        )
        self.assertFalse(success)
        self.assertIn("repository not found", message.lower())
        self.assertIn("unknown/repo", message)

    @patch('llm_context_os.utils.hf_downloader.hf_hub_download', side_effect=RevisionNotFoundError("Revision not found"))
    def test_download_revision_not_found_single_file(self, mock_hf_download):
        success, message = download_model_from_hf(
            repo_id="test/model-rev",
            target_dir=self.test_dir,
            filename="file.txt",
            revision="bad-revision"
        )
        self.assertFalse(success)
        self.assertIn("revision 'bad-revision' not found", message.lower())

    @patch('llm_context_os.utils.hf_downloader.snapshot_download', side_effect=RevisionNotFoundError("Revision not found"))
    def test_download_revision_not_found_snapshot(self, mock_snapshot_download):
        success, message = download_model_from_hf(
            repo_id="test/model-rev-snap",
            target_dir=self.test_dir,
            revision="bad-revision-snap"
        )
        self.assertFalse(success)
        self.assertIn("revision 'bad-revision-snap' not found", message.lower())


    @patch('llm_context_os.utils.hf_downloader.hf_hub_download', side_effect=HfHubHTTPError("HTTP error"))
    def test_download_http_error(self, mock_hf_download):
        success, message = download_model_from_hf(
            repo_id="test/http-error",
            target_dir=self.test_dir,
            filename="somefile.json"
        )
        self.assertFalse(success)
        self.assertIn("http error accessing file", message.lower())

    @patch('llm_context_os.utils.hf_downloader.shutil.copy', side_effect=Exception("Disk full"))
    @patch('llm_context_os.utils.hf_downloader.hf_hub_download')
    def test_download_general_exception_on_copy(self, mock_hf_download, mock_shutil_copy):
        mock_hf_download.return_value = str(self.dummy_cache_file)
        success, message = download_model_from_hf(
            repo_id="test/general-error",
            target_dir=self.test_dir,
            filename="another.gguf"
        )
        self.assertFalse(success)
        self.assertIn("unexpected error occurred", message.lower())
        self.assertIn("disk full", message.lower())

    @patch('llm_context_os.utils.hf_downloader.hf_hub_download', side_effect=Exception("Very generic error"))
    def test_download_general_exception_on_hf_call(self, mock_hf_download):
        success, message = download_model_from_hf(
            repo_id="test/generic-hf-error",
            target_dir=self.test_dir,
            filename="config.json"
        )
        self.assertFalse(success)
        self.assertIn("unexpected error occurred", message.lower())
        self.assertIn("very generic error", message.lower())

    # Test for the initial check if huggingface_hub is not available
    @patch('llm_context_os.utils.hf_downloader.HUGGINGFACE_HUB_AVAILABLE', False)
    def test_huggingface_hub_not_available(self):
        success, message = download_model_from_hf(
            repo_id="test/any",
            target_dir=self.test_dir,
            filename="any.file"
        )
        self.assertFalse(success)
        self.assertIn("huggingface_hub library or specific exceptions are not available", message)

if __name__ == '__main__':
    unittest.main()
