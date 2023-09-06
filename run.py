import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import csv
from vne.vae import ShapeVAE, ShapeSimilarityLoss
from vne.special.affinity_mat_create import similarity_matrix
from vne.special.alphanumeric_simulator import  alpha_num_Simulator
from vne.vis import plot_affinity, plot_loss,plot_umap, to_img
from vne.dataset import alphanumDataset, SubTomogram_dataset, CustomMNIST

from tqdm import tqdm
import umap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



alpha_num_list = "aebdijkz2uv"
aff_mat = "affinity.csv" #None
data_nat = "mnist" #'alphanum'
classes = 'classes.csv'
BATCH_SIZE = 128
LATENT_DIMS = 16
LEARNING_RATE = 1e-2
KLD_WEIGHT = 1. / (64*64)
BETA_FACT = 4
BETA = BETA_FACT * KLD_WEIGHT
GAMMA = 10
POSE_DIMS = 1
EPOCHS =50



with open(classes, newline='') as molecule_list_file:
    molecule_list = list(csv.reader(molecule_list_file, delimiter=','))[0]


if aff_mat is None and data_nat == 'alphanum':
    simulator = alpha_num_Simulator()
    lookup, imgs = similarity_matrix(simulator)

elif aff_mat:
    lookup =  np.genfromtxt(aff_mat, delimiter=',')


plot_affinity(lookup,molecule_list)



if data_nat == "mnist":
    dataset = CustomMNIST(root='./data', train=True)
    test_dataset = CustomMNIST(root='./data', train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
elif data_nat=="subtomo":
    dataset = SubTomogram_dataset(subtomo_path,IMAGES_PER_EPOCH, molecule_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last= True)
    molecule_list = dataset.keys()
elif data_nat=="alphanum":
    dataset = alphanumDataset(-45,45, list(alpha_num_list), simulator)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)






x = dataset[0]
fig = plt.figure()
fig.colorbar(plt.imshow(np.squeeze(x[0].numpy())))
#plt.imshow(to_img(x))
fig.savefig('data_before_loss_calc.png', dpi=144)

reconstruction_loss = nn.MSELoss() #Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input 
similarity_loss = ShapeSimilarityLoss(lookup=torch.Tensor(lookup).to(device))

model = ShapeVAE(
    latent_dims=LATENT_DIMS,
    pose_dims=POSE_DIMS,
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=1e-5)

loss_plot =[]
# The loss was not converging when weight_decay = 10^-2 
for epoch in range(EPOCHS):
    total_loss = 0
    for data in dataloader:
        img, mol_id = data

        img = Variable(img).to(device)
        mol_id = Variable(mol_id).to(device)
        # ===================forward=====================
        output, z, z_pose, mu, log_var = model(img)
        
        # reconstruction loss
        r_loss = reconstruction_loss(output, img)
        
        # kl loss 
        # https://arxiv.org/abs/1312.6114 (equation 10 ) 
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        # similarity loss
        s_loss = similarity_loss(mol_id, mu)
        
        loss = r_loss + (GAMMA * s_loss) + (BETA * kld_loss)      
        # ===================backward====================
        optimizer.zero_grad() # set the gradient of all optimised torch.tensors to zero
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    # ===================log========================
    loss_plot.append(total_loss.cpu().clone().numpy())
    print(f"epoch [{epoch+1}/{EPOCHS}], loss:{total_loss:.4f}, {r_loss.data}, {s_loss.data}, {kld_loss.data}")
    if epoch % 10 == 0 or epoch == EPOCHS-1:
        pic = to_img(output.to(device).data)
        save_image(pic, './image_{}.png'.format(epoch))


        enc = []
        lbl = []
        with torch.inference_mode():
            for i in tqdm(range(1000)):
                j = np.random.choice(range(len(dataset)))
                img, img_id= dataset[j]
                mu, log_var, pose = model.encode(img[np.newaxis,...].to(device))
                z = model.reparameterise(mu, log_var)
                enc.append(z.cpu())
                lbl.append(img_id)



        plot_umap(enc, lbl,epoch,molecule_list)
        plot_loss(loss_plot)

torch.save(model.state_dict(), './conv_autoencoder.pth')

# plot loss vs EPOCH
