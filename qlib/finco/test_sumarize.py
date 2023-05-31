import unittest
from dotenv import load_dotenv

from qlib.finco.task import SummarizeTask
from qlib.finco.workflow import WorkflowContextManager

load_dotenv(verbose=True, override=True)


class TestSummarize(unittest.TestCase):

    def test_execution(self):
        task = SummarizeTask()
        task.assign_context_manager(WorkflowContextManager())
        resp = task.execution()
        print(resp)

    def test_parse2txt(self):
        task = SummarizeTask()
        resp = task.parse2txt('')
        print(resp)


if __name__ == '__main__':
    unittest.main()
