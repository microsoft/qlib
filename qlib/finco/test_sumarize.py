import unittest

from qlib.finco.task import SummarizeTask


class TestSummarize(unittest.TestCase):
    def test_parse2txt(self):
        task = SummarizeTask()
        resp = task.parse2txt('')
        print(resp)


if __name__ == '__main__':
    unittest.main()
