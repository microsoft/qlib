import fire
from qlib.finco.workflow import WorkflowManager
from dotenv import load_dotenv
from qlib import auto_init


def main(prompt=None):
    load_dotenv(verbose=True, override=True)
    wm = WorkflowManager()
    wm.run(prompt)


if __name__ == "__main__":
    auto_init()
    fire.Fire(main)
