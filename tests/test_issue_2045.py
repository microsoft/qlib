import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from qlib.workflow.online.strategy import RollingStrategy

class TestIssue2045(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import qlib
        from pathlib import Path
        import shutil
        import os
        
        # Create dummy data for init
        cls.dummy_dir = Path("dummy_data_2045")
        if cls.dummy_dir.exists():
            shutil.rmtree(cls.dummy_dir)
        (cls.dummy_dir / "calendars").mkdir(parents=True)
        (cls.dummy_dir / "instruments").mkdir(parents=True)
        (cls.dummy_dir / "features").mkdir(parents=True)
        
        with open(cls.dummy_dir / "calendars" / "day.txt", "w") as f:
            f.write("2020-01-01\n2020-01-02\n")
            
        # Initialize
        qlib.init(provider_uri={"day": str(cls.dummy_dir.absolute())})

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.dummy_dir.exists():
            shutil.rmtree(cls.dummy_dir)

    def test_prepare_tasks_interval_check(self):
        # Mock dependencies
        # Create a mock that looks like RollingGen for the instance check
        from qlib.workflow.task.gen import RollingGen
        mock_rg = MagicMock(spec=RollingGen)
        mock_rg.step = 10  # Rolling step is 10
        mock_rg.gen_following_tasks.return_value = [{"task": "new_task"}]
        
        # Mock TimeAdjuster to return an interval smaller than step
        # We'll patch the class used inside RollingStrategy
        with patch("qlib.workflow.online.strategy.TimeAdjuster") as MockTimeAdjuster:
            mock_ta_instance = MockTimeAdjuster.return_value
            mock_ta_instance.cal_interval.return_value = 5  # Interval (5) < Step (10)
            
            # Mock OnlineToolR (attached to strategy)
            # We need to patch where it is imported or used. 
            # In strategy.py: self.tool = OnlineToolR(self.exp_name)
            with patch("qlib.workflow.online.strategy.OnlineToolR") as MockOnlineToolR:
                mock_tool_instance = MockOnlineToolR.return_value
                
                # Setup mock recorder
                mock_recorder = MagicMock()
                # Mock task config structure: task["dataset"]["kwargs"]["segments"]["test"] -> (start, end)
                # max_test will be the max of these tuples.
                mock_recorder.load_object.return_value = {
                    "dataset": {
                        "kwargs": {
                            "segments": {
                                "test": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-10"))
                            }
                        }
                    }
                }
                
                mock_tool_instance.online_models.return_value = [mock_recorder]
                
                # Instantiate strategy
                strategy = RollingStrategy(name_id="test_exp", task_template={}, rolling_gen=mock_rg)
                
                # Replace the internal tool/ta with our detailed mocks if needed, 
                # but patch should have handled the initialization if we did it right.
                # However, RollingStrategy.__init__ calls TimeAdjuster(), so mocking class works.
                # Same for OnlineToolR.
                
                # Call prepare_tasks
                # cur_time doesn't matter much because we mocked cal_interval, 
                # BUT transform_end_date is called on it.
                cur_time = pd.Timestamp("2021-01-15") 
                
                # EXECUTE
                tasks = strategy.prepare_tasks(cur_time)
                
                # VERIFY
                # Expected behavior (Fix): Should return [] because 5 < 10.
                self.assertEqual(len(tasks), 0, "Should NOT generate tasks when interval < step")

    def test_prepare_tasks_normal(self):
        # Mock dependencies
        from qlib.workflow.task.gen import RollingGen
        mock_rg = MagicMock(spec=RollingGen)
        mock_rg.step = 10
        mock_rg.gen_following_tasks.return_value = [{"task": "new_task"}]
        
        with patch("qlib.workflow.online.strategy.TimeAdjuster") as MockTimeAdjuster:
            mock_ta_instance = MockTimeAdjuster.return_value
            mock_ta_instance.cal_interval.return_value = 15  # Interval (15) > Step (10)
            
            with patch("qlib.workflow.online.strategy.OnlineToolR") as MockOnlineToolR:
                mock_tool_instance = MockOnlineToolR.return_value
                mock_recorder = MagicMock()
                mock_recorder.load_object.return_value = {
                    "dataset": { "kwargs": { "segments": { "test": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-10")) } } }
                }
                mock_tool_instance.online_models.return_value = [mock_recorder]
                
                strategy = RollingStrategy(name_id="test_exp", task_template={}, rolling_gen=mock_rg)
                
                cur_time = pd.Timestamp("2021-01-25")
                tasks = strategy.prepare_tasks(cur_time)
                
                self.assertEqual(len(tasks), 1, "Should generate tasks when interval > step")


if __name__ == "__main__":
    unittest.main()
