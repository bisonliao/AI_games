import unittest
import numpy as np
import torch

from octo.model.components.attention import AttentionRule
from octo.model.components.block_transformer_pt import (BlockTransformerPt, PrefixGroupPt,
                                                       TimestepGroupPt, TokenMetadataPt)


class AttentionTests(unittest.TestCase):
    def test_vectorized_matches_legacy_token_loop(self):
        prefix = [PrefixGroupPt(torch.zeros(2, 2, 8), torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
                                "task", {"*": AttentionRule.CAUSAL})]
        steps = [TimestepGroupPt(torch.zeros(2, 2, 3, 8), torch.ones(2, 2, 3, dtype=torch.bool),
                                "obs", {"*": AttentionRule.CAUSAL}),
                 TimestepGroupPt(torch.zeros(2, 2, 1, 8), torch.ones(2, 2, 1, dtype=torch.bool),
                                "readout", {"task": AttentionRule.ALL, "obs": AttentionRule.STRICT_PAST,
                                            "readout": AttentionRule.CURRENT})]
        for correct in [True, False]:
            for prefixes in [prefix, []]:
                block = BlockTransformerPt(transformer_kwargs=dict(token_embedding_size=8, num_layers=1,
                                         num_attention_heads=2, mlp_dim=16), use_correct_attention=correct,
                                         enforce_causal=False)
                side = "right" if correct else "left"
                prefix_size = 2 if prefixes else 0
                metadata = []
                for index in range(prefix_size + 8):
                    if index < prefix_size:
                        group = prefixes[int(np.searchsorted([2], index, side=side))]
                        metadata.append(TokenMetadataPt.create(group, -1))
                    else:
                        timestep, offset = divmod(index - prefix_size, 4)
                        group = steps[int(np.searchsorted([3, 4], offset, side=side))]
                        metadata.append(TokenMetadataPt.create(group, timestep))
                reference = torch.tensor([[query.should_attend_to(key) for key in metadata] for query in metadata])
                reference = reference & block.generate_pad_attention_mask(prefixes, steps)
                actual = block.generate_attention_mask(prefixes, steps, False)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
