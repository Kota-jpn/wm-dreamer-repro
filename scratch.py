import sys
sys.path.insert(0, './src')
import torch
from dreamer.world_model import WorldModel
from tests.test_world_model import make_wm, make_batch
wm = make_wm()
batch = make_batch(B=2, T=5)
total, metrics, _ = wm.compute_loss(batch)
print(f"KL value before backward: {metrics['wm/kl']}")
total.backward()
for name, p in wm.named_parameters():
    if p.grad is None or p.grad.abs().sum() == 0:
        print(f"No grad: {name}")
