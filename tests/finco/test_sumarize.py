import unittest
from dotenv import load_dotenv
# pydantic support load_dotenv,   so load_dotenv will be deprecated in the future.

from qlib.finco.task import SummarizeTask
from qlib.finco.workflow import WorkflowContextManager
from qlib.finco.llm import try_create_chat_completion

load_dotenv(verbose=True, override=True)


class TestSummarize(unittest.TestCase):

    def test_chat(self):
        messages = [
            {
                "role": "system",
                "content": "Your are a professional financial assistant.",
            },
            {
                "role": "user",
                "content": "How to write a perfect quant strategy.",
            },
        ]
        response = try_create_chat_completion(messages=messages)
        print(response)

    def test_execution(self):
        task = SummarizeTask()
        context = WorkflowContextManager()
        context.set_context("output_path", "../../examples/benchmarks/Linear")
        context.set_context("user_prompt", "My main focus is on the performance of the strategy's return."
                                           "Please summarize the information and give me some advice.")
        task.assign_context_manager(context)
        resp = task.execution()
        print(resp)

    def test_parse2txt(self):
        task = SummarizeTask()
        resp = task.get_info_from_file('')
        print(resp)


if __name__ == '__main__':
    unittest.main()
