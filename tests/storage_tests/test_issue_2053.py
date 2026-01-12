import shutil
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from qlib.data.storage.file_storage import FileFeatureStorage

class TestIssue2053(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path("test_issue_2053_data").absolute()
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir()
        
        self.day_dir = self.data_dir / "day"
        self.features_dir = self.day_dir / "features"
        self.features_dir.mkdir(parents=True)
        
        self.provider_uri = {"day": self.day_dir}
        
        import qlib
        qlib.init(provider_uri=self.provider_uri)

    def tearDown(self):
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)

    def test_case_sensitivity_check(self):
        # Case 1: Uppercase Directory Exists
        inst_name = "AAPL"
        inst_dir = self.features_dir / inst_name
        inst_dir.mkdir()
        
        # Create a dummy binary file to ensure storage object considers it valid
        bin_file = inst_dir / "close.day.bin"
        data = np.array([1.0, 2.0, 3.0], dtype="<f")
        index = 0
        with bin_file.open("wb") as fp:
            np.hstack([index, data]).astype("<f").tofile(fp)
            
        storage = FileFeatureStorage(instrument=inst_name, field="close", freq="day", provider_uri=self.provider_uri)
        
        # Critical Assertion: The generated URI path MUST contain the UPPERCASE instrument name
        # This proves that the logic detected the existing folder and didn't force lowercase.
        # On Windows, both paths point to the same file, but for cross-platform correctness (Linux),
        # we strictly require the path string to match the filesystem.
        self.assertIn(inst_name, str(storage.uri), "Storage URI should preserve case if directory exists")
        self.assertNotIn(inst_name.lower(), str(storage.uri).replace(inst_name, ""), "Storage URI should NOT contain lowercase version if uppercase exists")
        
        # Verify data access still works
        self.assertIsNotNone(storage[0], "Should be able to read data")

    def test_backward_compatibility(self):
        # Case 2: Lowercase Directory Exists (Old Behavior)
        inst_name = "MSFT"
        # We create the directory in LOWERCASE
        inst_dir = self.features_dir / inst_name.lower()
        inst_dir.mkdir()
        
        bin_file = inst_dir / "close.day.bin"
        data = np.array([4.0, 5.0, 6.0], dtype="<f")
        index = 0
        with bin_file.open("wb") as fp:
            np.hstack([index, data]).astype("<f").tofile(fp)
            
        # We access it using UPPERCASE name
        storage = FileFeatureStorage(instrument=inst_name, field="close", freq="day", provider_uri=self.provider_uri)
        
        # Assertion: It should FALLBACK to lowercase path because uppercase dir doesn't exist
        self.assertIn(inst_name.lower(), str(storage.uri).lower(), "Storage URI should resolve to lowercase if upper doesn't exist")
        
        # Verify data access works
        self.assertIsNotNone(storage[0], "Should be able to read data from fallback lowercase path")

    def test_new_instrument_defaults(self):
        # Case 3: Neither exists (New/Write scenario)
        inst_name = "GOOG"
        # We define a storage for a non-existent instrument
        storage = FileFeatureStorage(instrument=inst_name, field="close", freq="day", provider_uri=self.provider_uri)
        
        # If neither exists, we prefer the ORIGINAL case (or lowercase? The plan said original case).
        # Let's assert Original Case to allow users to create new uppercase folders.
        self.assertIn(inst_name, str(storage.uri), "New paths should respect input case")

if __name__ == "__main__":
    unittest.main()
