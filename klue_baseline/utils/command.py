from typing import List


class Command:

    Train = "train"
    Evaluate = "evaluate"
    Test = "test"

    @staticmethod
    def tolist() -> List[str]:
        return [Command.Train, Command.Evaluate, Command.Test]
