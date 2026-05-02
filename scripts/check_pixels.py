from dm_control import suite
import numpy as np

env = suite.load('walker', 'walk', task_kwargs={'random': 0})
ts = env.reset()

  # pixels を取る
pixels = env.physics.render(height=64, width=64, camera_id=0)
print("pixel shape:", pixels.shape, "dtype:", pixels.dtype)
print("min/max:", pixels.min(), pixels.max())

# 保存
import imageio
imageio.imwrite('results/sample_frame.png', pixels)
print("Saved to results/sample_frame.png")