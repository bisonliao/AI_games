def shaped_reward(
    before,
    after,
    event=None,
    terminal=False,
    outcome=None,
    cfg=None,
    before_visible_counts=None,
    after_visible_counts=None,
    shaping_delta=None,
):
    scale = getattr(cfg, 'potential_scale', 0.5)
    # Potential shaping is intentionally emitted only at the completed
    # discard boundary.  Other actions (peng/kong/pass/win), and the first
    # discard before a previous baseline exists, contribute no shaping term.
    r = 0.0 if shaping_delta is None else float(shaping_delta) * scale
    if event:
        r += getattr(cfg, 'event_reward', 0.0)
    if terminal:
        terminal_rewards = {
            'win': getattr(cfg, 'win_reward', 10.0),
            'loss': getattr(cfg, 'loss_reward', -10.0),
            'draw': getattr(cfg, 'draw_reward', -10.0),
        }
        r += terminal_rewards.get(outcome, 0.0)
    return float(r)
