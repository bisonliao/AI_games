import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from robot_arm.octo.runtime import (CHECKPOINT_DIR, load_checkpoint, save_checkpoint,
                                   TorchOnlyImports)
from robot_arm.octo.evaluate import predict


class NativeTests(unittest.TestCase):
    def test_backend_guard(self):
        guard = TorchOnlyImports()
        sys.meta_path.insert(0, guard)
        try:
            for module in ["jax", "flax", "tensorflow"]:
                self.assertNotIn(module, sys.modules)
                with self.assertRaisesRegex(RuntimeError, "Torch-only verification"):
                    importlib.import_module(module)
                self.assertNotIn(module, sys.modules)
        finally:
            sys.meta_path.remove(guard)

    @unittest.skipUnless((CHECKPOINT_DIR / "base.pt").exists(), "Run convert to validate the native checkpoint")
    def test_native_roundtrip_and_frozen_text(self):
        torch.set_num_threads(4)
        model, payload = load_checkpoint(CHECKPOINT_DIR / "base.pt")
        model.eval()
        text = [parameter for name, parameter in model.named_parameters() if ".hf_model." in name]
        self.assertTrue(text)
        self.assertTrue(all(not parameter.requires_grad for parameter in text))
        with torch.inference_mode():
            expected = predict(model, model.example_batch["observation"], payload["metadata"])
        self.assertEqual(expected.shape, (1, 4))
        self.assertTrue(torch.isfinite(expected).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, payload["metadata"])
            restored, restored_payload = load_checkpoint(path)
            restored.eval()
            with torch.inference_mode():
                actual = predict(restored, restored.example_batch["observation"], restored_payload["metadata"])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for backend in ["jax", "flax", "tensorflow"]:
            self.assertNotIn(backend, sys.modules)


if __name__ == "__main__":
    unittest.main()
