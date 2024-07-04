import unittest
from qlib.finco.utils import SingletonBaseClass


class SingletonTest(unittest.TestCase):

    def test_singleton(self):
        # self.assertEqual(self.to_str(data.tail()), self.to_str(res))
        closure_checker = []

        class A(SingletonBaseClass):

            def __init__(self) -> None:
                closure_checker.append(0)

        A()
        self.assertEqual(len(closure_checker), 1)
        A()
        self.assertEqual(len(closure_checker), 1)


if __name__ == "__main__":
    unittest.main()
