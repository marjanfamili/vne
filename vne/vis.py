import numpy as np
import matplotlib.pyplot as plt
import umap
import random
import torch
from skimage.util import montage

def plot_affinity(lookup,molecule_list):

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(lookup, vmin=-1, vmax=1, cmap=plt.cm.get_cmap('RdBu'))
    ax.set_title('Shape similarity matrix')
    ax.set_xticks(np.arange(0, len(molecule_list)))
    ax.set_xticklabels(molecule_list)
    ax.set_yticks(np.arange(0, len(molecule_list)))
    ax.set_yticklabels(molecule_list)
    ax.tick_params(axis='x', rotation=90)
    fig.colorbar(im, ax=ax)
    plt.savefig('similarity.png', dpi=144)



def plot_loss(loss_plot,  kldloss_plot,sloss_plot, rloss_plot):
    plt.figure(figsize=(16,16))
    plt.plot(loss_plot, label="Total Loss", linewidth=3)
    plt.plot(kldloss_plot, label="KLD Loss", linewidth=3)
    plt.plot(sloss_plot, label="Similarity Loss", linewidth=3)
    plt.plot(rloss_plot, label="reconstruction Loss", linewidth=3)
    plt.yscale("log")
    plt.xlabel('EPOCHS')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_vs_EPOCHS.png", dpi=144)


def plot_umap(enc,lbl,epoch,  molecule_list, filename):
    enc = np.concatenate(enc, axis=0)
    plt.clf()
    plt.plot(enc)
    plt.savefig("latent_dist.png")
    plt.clf()
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(enc)


    fig, ax = plt.subplots(figsize=(16, 16))
    for mol_id, mol in enumerate(molecule_list):
        idx = np.where(np.array(lbl) == mol_id)[0]
        cmap = plt.cm.get_cmap("tab20")
        color = cmap(mol_id % 20)
        
        scatter = ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s = 64,
            label = mol[:4],
            facecolor=color,
            edgecolor=color,
        )
        
    #     ax.plot(x_enc[:, 0], x_enc[:, 1], 'ko', markersize=42)
    ax.legend()
    ax.set_title(f'UMAP projection', fontsize=24)
    plt.savefig(filename, dpi=144)

def plot_z_disentanglement(dataset,model,device):
    enc = []
    for num_fig in range(5):
        draw_four = random.sample(range(len(dataset)), k=4)
        for i in range(4):
            img, img_id= dataset[i]
            mu, log_var, pose = model.encode(img[np.newaxis,...].to(device))
            z = model.reparameterise(mu, log_var)
            enc.append(z.cpu())

    
        # Number of interpolation steps
        num_steps = 10
        fig, axes = plt.subplots(num_steps, num_steps, figsize=(num_steps*2, num_steps*2))

        for i in range(num_steps):
            for j in range(num_steps):
                t1, t2 = i /(num_steps-1), j / (num_steps-1)


                # Linear interpolation in latent space
                interpolated_encoding = (1 - t1) * ((1 - t2) * enc[0] + t2 * enc[1]) + t1 * ((1 - t2) * enc[2] + t2 * enc[3])

                # Decode the interpolated encoding to generate an image
                with torch.no_grad():
                    decoded_image = model.decode(interpolated_encoding.to(device), torch.Tensor(1, 1))
                
                axes[i, j].imshow(decoded_image.cpu().squeeze().numpy(), cmap='gray')
                axes[i, j].axis('off')


    
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0, hspace=0)

        # Save the figure
        plt.savefig("z_interpolate_{num_fig}.png")


def plot_pose_interpolation(model,dataset,device):
    with torch.inference_mode():
        r = []
        y = dataset[0]
        x, z, z_pose, mu, logvar = model(y[0][np.newaxis, ...].to(device=device, dtype=torch.float))

        for theta in np.linspace(1, 1, 16):
            y_hat = model.decode(z.to(device), torch.tensor([[float(theta)]]))
            r.append(np.squeeze(y_hat[0].cpu()))
    r = np.stack(r, axis=0)
    m = montage(r, grid_shape=(1, r.shape[0]))
    plt.figure(figsize=(16, 2))
    plt.imshow(m)
    plt.savefig("pose_interpolation.png", dpi=144)


def plot_pose(model,dataset,device):
    pose_angle = []
    r = []
    THETA_LOWER = -45
    THETA_UPPER = 45

    with torch.inference_mode():
        for theta in range(THETA_LOWER,THETA_UPPER):
            # This is at the moement not working because you need to take 
            # an image at zero degree rotation and then rotate that by specific angle theta
            # and then do the following
            y = dataset[3]
            x, z, z_pose, mu, logvar = model(y[0][np.newaxis, ...].to(device=device))     
            if theta % 5 == 0 : 
                r.append(np.squeeze(x[0,0,...].cpu()))
            pose_angle.append(z_pose[0,0].cpu().clone().numpy())

    fig, ax = plt.subplots(2,1, figsize=(10,15))
    ax1, ax2 = ax


    coef = np.polyfit(range(THETA_LOWER,THETA_UPPER),np.squeeze(pose_angle),1)
    poly1d_fn = np.poly1d(coef) 
    ax1.plot(range(THETA_LOWER,THETA_UPPER),np.squeeze(pose_angle),'yo',range(THETA_LOWER,THETA_UPPER),poly1d_fn(range(THETA_LOWER,THETA_UPPER)),'--k',
        linewidth=2)
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel('pose')

    r = np.stack(r, axis=0)
    m = montage(r, grid_shape=(3, 6))
    ax2.imshow(m)
    ax2.axis('off')

    fig.tight_layout()
    fig.savefig("pose_angle_images.png", dpi=144)

    
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1) #. Clamps all elements in input into the range [ min, max ]. 
    x = x.view(x.size(0), 1, 64, 64) # Returns a new tensor with the same data as the self tensor but of a different shape.
    return x
