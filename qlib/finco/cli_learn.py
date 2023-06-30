import fire
from qlib.finco.workflow import LearnManager
from dotenv import load_dotenv
from qlib import auto_init


def main(prompt=None):
    load_dotenv(verbose=True, override=True)
    lm = LearnManager()
    lm.run(prompt)


if __name__ == "__main__":
    auto_init()
    fire.Fire(main)
