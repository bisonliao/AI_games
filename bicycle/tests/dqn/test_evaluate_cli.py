"""Command-line parsing tests for optional evaluation GUI display."""

from dqn.evaluate import build_parser


def test_display_is_disabled_by_default():
    args = build_parser().parse_args(["--checkpoint", "model.pt"])
    assert args.display is False
    assert args.env_id == 1


def test_display_flag_enables_gui():
    args = build_parser().parse_args(["--checkpoint", "model.pt", "--display"])
    assert args.display is True


def test_environment_two_can_be_selected():
    args = build_parser().parse_args(
        ["--checkpoint", "model.pt", "--env_id", "2"]
    )
    assert args.env_id == 2
