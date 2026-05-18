from ctta.config import parse_args


def main():
    args = parse_args()
    # Initialize CUDA before importing the heavier CTTA stack. In this
    # environment, delayed initialization can make transformers' bf16 check
    # incorrectly report that GPU/bf16 is unavailable after model imports.
    import torch

    torch.cuda.is_available()
    from ctta.engine import CTTAEngine

    CTTAEngine(args).run()


if __name__ == "__main__":
    main()
