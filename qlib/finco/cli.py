import fire
from qlib.finco.task import WorkflowManager
from dotenv import load_dotenv


def main(prompt=None):
    load_dotenv(verbose=True, override=True)
    wm = WorkflowManager()
    wm.run(prompt)


if __name__ == "__main__":
    fire.Fire(main)
