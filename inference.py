import torch
import numpy as np
from net.conv import Conv
from PIL import Image
import matplotlib.pyplot as plt

model = Conv.load_from_checkpoint("./models/state_dict/0615120754/epoch=4-step=30000.ckpt")
model.eval()
image = 1-np.array(Image.open('./sample'))
x = torch.Tensor(image)
x = x.unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    y_hat = model(x)
    y_hat = torch.softmax(y_hat, dim=1)
    max_ind = int(torch.argmax(y_hat))
    plt.imshow(image, cmap='gray')
    plt.title(f'Guess is {max_ind} with {int(np.squeeze(y_hat.cpu().numpy())[max_ind]*100)}% confidence')
    plt.show()