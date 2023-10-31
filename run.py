import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import csv
from vne.vae import ShapeVAE, ShapeSimilarityLoss
from vne.special.affinity_mat_create import similarity_matrix
from vne.special.alphanumeric_simulator import  alpha_num_Simulator
from vne.vis import plot_affinity, plot_loss,plot_umap, to_img, plot_pose, plot_z_disentanglement
from vne.dataset import alphanumDataset, SubTomogram_dataset, CustomMNIST
from vne.read_config import get_config_values
from tqdm import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a command line argument parser
parser = argparse.ArgumentParser(description="Read and process a YAML configuration file.")
parser.add_argument("--config_file", required=True, help="Path to the YAML configuration file")

# Parse the command line arguments
args = parser.parse_args()
# Read the YAML file and store the variables
yaml_file_path = args.config_file
config_data = get_config_values(yaml_file_path)


LATENT_DIMS = config_data.get('latent_dims')
POSE_DIMS = config_data.get('pose_dims')
EPOCHS = config_data.get('epochs')
BATCH_SIZE = config_data.get('batch_size')
LEARNING_RATE = config_data.get('learning_rate')
alpha_num_list = config_data.get('alpha_num_list')
data_nat = config_data.get('data_nature')
classes = config_data.get('classes')
aff_mat = config_data.get('affinity')
datapath = config_data.get('datapath')
datapath_test = config_data.get('datapath_test')
GAMMA = config_data.get('gamma')
BETA_FACT = config_data.get('beta_fact')
data_format = config_data.get('data_format')
IMAGES_PER_EPOCH=10000
KLD_WEIGHT = 1. / (64*64)
BETA = BETA_FACT * KLD_WEIGHT
print(data_format, BETA_FACT, GAMMA, datapath, aff_mat, classes, data_nat, alpha_num_list, BATCH_SIZE, EPOCHS, POSE_DIMS, LATENT_DIMS)


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
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
elif data_nat=="subtomo":
    dataset = SubTomogram_dataset(datapath,IMAGES_PER_EPOCH, molecule_list,data_format)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = SubTomogram_dataset(datapath_test,IMAGES_PER_EPOCH, molecule_list,data_format)
    test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    molecule_list = dataset.keys()
elif data_nat=="alphanum":
    dataset = alphanumDataset(-45,45, list(alpha_num_list),IMAGES_PER_EPOCH, alpha_num_Simulator)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = alphanumDataset(-45,45, list(alpha_num_list+"v"),IMAGES_PER_EPOCH, alpha_num_Simulator)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

else:
    print("didnt define data_nat")

x = dataset[15]
print("data shape is : ", x[0].shape)
if not data_format=='mrc':
    fig = plt.figure()
    fig.colorbar(plt.imshow(np.squeeze(x[0].numpy())))
    #plt.imshow(to_img(x))
    fig.savefig('data_before_loss_calc.png', dpi=144)

reconstruction_loss = nn.MSELoss() #Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input 
similarity_loss = ShapeSimilarityLoss(lookup=torch.Tensor(lookup).to(device))

dims =dataset[0][0].shape[1:]
model = ShapeVAE(
    latent_dims = LATENT_DIMS,
    pose_dims = POSE_DIMS,
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=1e-5)

loss_plot = []
kldloss_plot = []
sloss_plot = []
rloss_plot = []
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
    kldloss_plot.append(BETA*kld_loss.cpu().clone().detach().numpy())
    sloss_plot.append(GAMMA*s_loss.cpu().clone().detach().numpy())
    rloss_plot.append(r_loss.cpu().clone().detach().numpy())

    print(f"epoch [{epoch+1}/{EPOCHS}], loss:{total_loss:.4f}, reconstruction : {r_loss.data}, Affinity: {s_loss.data}, KLD: {kld_loss.data}")

    if epoch % 10 == 0 or epoch == EPOCHS-1:

        if data_format!="mrc":

            pic = to_img(output.to(device).data)
            save_image(pic, './image_{}.png'.format(epoch))

        elif data_format=="mrc":
            mrcfile.write(f'./{epoch}_r{molecule_list[mol_id[0]]}.mrc', output[0,0,:,:,:].cpu().detach().numpy(), overwrite=True )
            mrcfile.write(f'./{epoch}_o{molecule_list[mol_id[0]]}.mrc', img[0,0,:,:,:].cpu().detach().numpy(), overwrite=True )

        enc = []
        enc_train = []
        lbl = []
        lbl_train = []
        with torch.inference_mode():
            for i in tqdm(range(1000)):
                j = np.random.choice(range(len(test_dataset)))
                img, img_id= test_dataset[j]
                mu, log_var, pose = model.encode(img[np.newaxis,...].to(device))
                z = model.reparameterise(mu, log_var)
                enc.append(z.cpu())
                lbl.append(img_id)
                k = np.random.choice(range(len(dataset)))
                img, img_id= dataset[k]
                mu, log_var, pose = model.encode(img[np.newaxis,...].to(device))
                z = model.reparameterise(mu, log_var)
                enc_train.append(z.cpu())
                lbl_train.append(img_id)


        plot_umap(enc, lbl,epoch,molecule_list, f"UMAP_test{epoch}")
        plot_umap(enc_train, lbl_train,epoch,molecule_list, f"UMAP_train{epoch}")
        plot_loss(loss_plot, kldloss_plot,sloss_plot,rloss_plot)

plot_z_disentanglement(dataset,model,device)

torch.save(model.state_dict(), './conv_autoencoder.pth')
plot_pose(model,dataset,device)
# plot loss vs EPOCH
