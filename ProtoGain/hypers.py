import json


class Params:

    def __init__(
        self,
        input=None,
        output=None,
        ref=None,
        output_folder=None,
        num_iterations=2000,
        batch_size=128,
        alpha=0.2,
        miss_rate=0.2,
        hint_rate=0.9,
        train_ratio=0.8,
        lr_D=0.001,
        lr_G=0.001,
        num_runs=1,
        override=0,
    ):
        self.input = input
        self.output = output
        self.output_folder = output_folder
        self.ref = ref
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.alpha = alpha
        self.miss_rate = miss_rate
        self.hint_rate = hint_rate
        self.train_ratio = train_ratio
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.num_runs = num_runs
        self.override = override

    @staticmethod
    def _read_json(json_file):
        with open(json_file, "r") as f:
            params = json.load(f)
        return params

    @classmethod
    def read_hyperparameters(cls, params_json=None):

        params = cls._read_json(params_json)

        print(params)

        input = params["input"]
        output = params["output"]
        ref = params["ref"]
        output_folder = params["output_folder"]
        num_iterations = params["num_iterations"]
        batch_size = params["batch_size"]
        alpha = params["alpha"]
        miss_rate = params["miss_rate"]
        hint_rate = params["hint_rate"]
        train_ratio = params["train_ratio"]
        lr_D = params["lr_D"]
        lr_G = params["lr_G"]
        num_iterations = params["num_iterations"]
        override = params["override"]

        return cls(
            input,
            output,
            ref,
            output_folder,
            num_iterations,
            batch_size,
            alpha,
            miss_rate,
            hint_rate,
            train_ratio,
            lr_D,
            lr_G,
            num_iterations,
            override,
        )

    def update_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
