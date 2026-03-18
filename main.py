from flpoison.cli.args import read_args
from flpoison.fl.training import main


if __name__ == "__main__":
    args, cli_args = read_args()
    main(args, cli_args)
