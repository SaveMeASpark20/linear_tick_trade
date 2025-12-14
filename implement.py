### Research

import models
import torch
import researches

model = models.LinearModel(3)
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
print(model.eval())


### AR(3) Model to predict  future log return - 12h forecast horizon
researches.print_model_params(model)

# ~14% without any optimization
# 1. Compounding Trade Sizing
# 2. Leverage
# ~14% to > 40 %

