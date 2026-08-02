from dqn.evaluate import build_parser


def test_display_is_disabled_by_default():
    args = build_parser().parse_args(["--checkpoint", "model.pt"])
    assert args.display is False


def test_display_flag_enables_gui():
    args = build_parser().parse_args(["--checkpoint", "model.pt", "--display"])
    assert args.display is True
