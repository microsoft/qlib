import unittest
import os
import shutil

from dotenv import load_dotenv
# pydantic support load_dotenv,   so load_dotenv will be deprecated in the future.

from qlib.finco.task import SummarizeTask
from qlib.finco.workflow import WorkflowContextManager
from qlib.finco.llm import APIBackend
from qlib.finco.workflow import WorkflowManager
from qlib.finco.knowledge import PracticeKnowledge, YamlStorage

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
        response = APIBackend().try_create_chat_completion(messages=messages)
        print(response)

    def test_execution(self):
        task = SummarizeTask()
        context = WorkflowContextManager()
        context.set_context("workspace", "../../examples/benchmarks/Linear")
        context.set_context("user_prompt", "My main focus is on the performance of the strategy's return."
                                           "Please summarize the information and give me some advice.")
        task.assign_context_manager(context)
        resp = task.execute()
        print(resp)

    def test_generate_batch_result(self):
        wm = WorkflowManager()

        prompt = wm.default_user_prompt
        # prompt = ""

        workdir = os.path.dirname(wm.get_context().get_context("workspace"))
        summaries_path = os.path.join(workdir, "summaries")

        if not os.path.exists(summaries_path):
            os.makedirs(summaries_path)

        for i in range(10):
            wm.run(prompt)
            if os.path.exists(f"{workdir}/finCoReport.md"):
                shutil.move(f"{workdir}/finCoReport.md", f"{workdir}/summaries/finCoReport{i}.md")

    def test_parse2txt(self):
        task = SummarizeTask()
        resp = task.get_info_from_file("")
        print(resp)

    def test_practice_knowledge(self):
        pk = PracticeKnowledge(YamlStorage(path.joinpath(Path.cwd().joinpath("knowledge")/f"{self.KT_PRACTICE}/{YamlStorage.DEFAULT_NAME}")))
        pk.add(["test1", "test2"])

if __name__ == "__main__":
    unittest.main()
