import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io
from download_models import download


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        # Keep test environments on disk, under the project.
        root = Path(__file__).resolve().parents[1] / '.local/test-downloads'
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name)
        self.payload = b'checkpoint-fixture'
        self.entry = dict(id='test', file='test.pt', bytes=len(self.payload),
                          sha256=hashlib.sha256(self.payload).hexdigest(), url='https://example.invalid/test.pt')

    def test_download_and_cached_verification(self):
        with patch('download_models.urllib.request.urlopen', return_value=io.BytesIO(self.payload)) as request:
            path = download(self.entry, self.dest)
            self.assertEqual(path.read_bytes(), self.payload)
            download(self.entry, self.dest)
            request.assert_called_once()

    def test_corrupt_download_is_not_installed(self):
        with patch('download_models.urllib.request.urlopen', return_value=io.BytesIO(b'x' * len(self.payload))):
            with self.assertRaises(ValueError):
                download(self.entry, self.dest)
        self.assertEqual(list(self.dest.iterdir()), [])

    def test_oversized_download_is_not_installed(self):
        with patch('download_models.urllib.request.urlopen', return_value=io.BytesIO(self.payload + b'x')):
            with self.assertRaises(ValueError):
                download(self.entry, self.dest)
        self.assertEqual(list(self.dest.iterdir()), [])

    def test_existing_unrelated_checkpoint_is_preserved(self):
        path = self.dest / 'test.pt'
        path.write_bytes(b'local training')
        with patch('download_models.urllib.request.urlopen') as request:
            with self.assertRaises(ValueError):
                download(self.entry, self.dest)
            request.assert_not_called()
        self.assertEqual(path.read_bytes(), b'local training')
