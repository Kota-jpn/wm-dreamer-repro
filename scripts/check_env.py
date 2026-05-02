from dm_control import suite                                                                                                                                                                              
import numpy as np                                              

env = suite.load('walker', 'walk')
ts = env.reset()
print("obs keys:", list(ts.observation.keys()))
print("orientations shape:", ts.observation['orientations'].shape)

action_spec = env.action_spec()
print("action shape:", action_spec.shape)
print("action min/max:", action_spec.minimum, action_spec.maximum)

# 5 step random rollout
for i in range(5):
      a = np.random.uniform(action_spec.minimum, action_spec.maximum)

action_spec = env.action_spec()
print("action shape:", action_spec.shape)
print("action min/max:", action_spec.minimum, action_spec.maximum)

# 5 step random rollout
for i in range(5):
    a = np.random.uniform(action_spec.minimum, action_spec.maximum)
    ts = env.step(a)
    print(f"step {i}: reward={ts.reward}, done={ts.last()}")

print("OK")
