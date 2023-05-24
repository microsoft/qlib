import fire
from qlib.finco.task import WorkflowManager


def main(prompt):
    wm = WorkflowManager()
    wm.run(prompt)


if __name__ == "__main__":
    fire.Fire(main)
