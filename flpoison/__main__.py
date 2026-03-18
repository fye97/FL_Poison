from flpoison.cli.args import read_args
from flpoison.fl.training import main as training_main


def main():
    args, cli_args = read_args()
    training_main(args, cli_args)


if __name__ == "__main__":
    main()
