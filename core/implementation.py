import models
import torch  

model = models.LinearModel(3)
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()


