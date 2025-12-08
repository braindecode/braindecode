import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from braindecode.datasets.hub import HubDatasetMixin
class MockDataset(HubDatasetMixin):
    def __init__(self, description):
        self.description = description
class TestHubBIDS(unittest.TestCase):
    def test_save_dataset_card_dataframe_with_bids(self):
        """Test _save_dataset_card with DataFrame description containing BIDS info."""
        # Create a dummy DataFrame with BIDS columns
        df = pd.DataFrame({
            "subject": ["1", "1", "2"],
            "session": ["A", "B", "A"],
            "task": ["rest", "rest", "rest"],
            "run": ["1", "2", "1"],
            "other": [1, 2, 3]
        })
        ds = MockDataset(df)
        
        # Mock _generate_readme_content to check what it receives
        with patch.object(ds, "_generate_readme_content") as mock_gen:
            with patch("builtins.open", new_callable=MagicMock) as mock_open:
                ds._save_dataset_card("dummy_path")
                
                # Check if _generate_readme_content was called with correct bids_info
                call_args = mock_gen.call_args
                if call_args:
                    _, kwargs = call_args
                    bids_info = kwargs.get("bids_info")
                    
                    self.assertIsNotNone(bids_info)
                    self.assertEqual(bids_info["subject"], 2) # 2 unique subjects
                    self.assertEqual(bids_info["session"], 2) # 2 unique sessions (A, B)
                    self.assertEqual(bids_info["task"], 1)    # 1 unique task
                    self.assertEqual(bids_info["run"], 2)     # 2 unique runs (1, 2)
    def test_save_dataset_card_series_with_bids(self):
        """Test _save_dataset_card with Series description containing BIDS info."""
        # Create a dummy Series with BIDS index
        series = pd.Series([1, "A", "rest"], index=["subject", "session", "task"])
        ds = MockDataset(series)
        
        with patch.object(ds, "_generate_readme_content") as mock_gen:
            with patch("builtins.open", new_callable=MagicMock) as mock_open:
                ds._save_dataset_card("dummy_path")
                
                call_args = mock_gen.call_args
                if call_args:
                    _, kwargs = call_args
                    bids_info = kwargs.get("bids_info")
                    
                    self.assertIsNotNone(bids_info)
                    self.assertEqual(bids_info["subject"], 1)
                    self.assertEqual(bids_info["session"], 1)
                    self.assertEqual(bids_info["task"], 1)
                    # 'run' is missing, should not be in bids_info
                    self.assertNotIn("run", bids_info)
    def test_save_dataset_card_no_bids(self):
        """Test _save_dataset_card with no BIDS info."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds = MockDataset(df)
        
        with patch.object(ds, "_generate_readme_content") as mock_gen:
            with patch("builtins.open", new_callable=MagicMock) as mock_open:
                ds._save_dataset_card("dummy_path")
                
                call_args = mock_gen.call_args
                if call_args:
                    _, kwargs = call_args
                    bids_info = kwargs.get("bids_info")
                    # Should be empty dict
                    self.assertEqual(bids_info, {})
if __name__ == "__main__":
    unittest.main()