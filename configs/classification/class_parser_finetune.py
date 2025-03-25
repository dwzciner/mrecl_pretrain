import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('--model-path', nargs='+', type=str, help='model path', default=None)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)