import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
# from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class ConvCVAE(nn.Module):
    def __init__(self, latent_size = 128, num_classes = 17, device = 'auto'): #for spectro 128,163
        super(ConvCVAE, self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.device = device

        # Encoder
        self.encoder_conv1 = nn.Conv2d(2 + num_classes, 32, kernel_size=4, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.encoder_conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        self.encoder_fc1 = nn.Linear(512 * 16 * 26, 100)
        self.encoder_fc_mu = nn.Linear(100, latent_size)
        self.encoder_fc_logvar = nn.Linear(100, latent_size)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_size + num_classes, 100)
        self.decoder_fc2 = nn.Linear(100, 512 * 16 * 26 )
        self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_conv5 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1, output_padding=(1,0))#, output_padding=(1,1)
        
        
    def encoder(self, x, labels):
        device = x.device
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        y_maps = y_onehot.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3]).to(device)
        x = torch.cat((x, y_maps), dim=1)
        x = F.leaky_relu(self.encoder_conv1(x))
        x = F.leaky_relu(self.encoder_conv2(x))
        x = F.leaky_relu(self.encoder_conv3(x))
        x = F.leaky_relu(self.encoder_conv4(x))
        x = F.leaky_relu(self.encoder_conv5(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.encoder_fc1(x))
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar


    def unFlatten(self, x):
        return x.reshape(x.shape[0], 512, 16, 26)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Automatically on correct device
        return mu + eps * std

    
    def decoder(self, z):
        # print("Decoder \n linear input shape:", z.shape)
        # sys.stdout.flush()
        z = F.leaky_relu(self.decoder_fc1(z))
        z = F.leaky_relu(self.decoder_fc2(z))
        # print("linear shape:", z.shape)
        # sys.stdout.flush()
        xdot = self.unFlatten(z)
        # print("Flatt:", z.shape)
        # sys.stdout.flush()
        xdot = F.leaky_relu(self.decoder_conv1(xdot))
        # print("Conv1 output shape:", xdot.shape)
        # sys.stdout.flush()
        xdot = F.leaky_relu(self.decoder_conv2(xdot))
        # print("Conv2 output shape:", xdot.shape)
        # sys.stdout.flush()
        xdot = F.leaky_relu(self.decoder_conv3(xdot))
        # print("Conv3 output shape:", xdot.shape)
        # sys.stdout.flush()
        xdot = F.leaky_relu(self.decoder_conv4(xdot))
        xdot = F.leaky_relu(self.decoder_conv5(xdot))
        # print("Conv4 output shape:", xdot.shape)
        # sys.stdout.flush()
        # z = F.leaky_relu(self.decoder_conv5(z))
        return xdot
    
    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        labels = F.one_hot(labels, num_classes=self.num_classes).float().to(z.device)
        z = torch.cat((z, labels), dim=1)
        pred = self.decoder(z)
        return pred, mu, logvar
        
    
    def loss_function(self, recon_x, x, mu, logvar, alpha = 1, beta = 1):
        BCE = F.mse_loss(recon_x, x, reduction='mean') 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return alpha * BCE + beta * KLD
    
    
    def sample(self, z, labels):
        device = z.device
        labels = labels.to(torch.long).to(device)
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(device)
        z = torch.cat((z, one_hot_labels), dim=1)
        return self.decoder(z)


class ConvCVAEPL(pl.LightningModule):
    # def __init__(self, latent_size = 128, num_classes = 163, device = 'auto'):
    def __init__(self, latent_size = 128, num_classes = 17, device = 'auto', learning_rate = 1e-3):
        super(ConvCVAEPL, self).__init__()
        self.model = ConvCVAE(latent_size, num_classes, device)
        self.learning_rate = learning_rate

    def on_train_epoch_start(self):
        # Slowly increase beta
        # ramp_period = 50
        # max_beta = 1
        # self.beta = torch.minimum(torch.exp(torch.tensor(0.1*(self.current_epoch-ramp_period))), torch.tensor(max_beta))

        ramp_period = 25
        amplitude = 0.15
        self.beta = float(1 / (1 + torch.exp(torch.tensor(-amplitude * (self.current_epoch - ramp_period)))))
        self.print(f"Beta = {self.beta:.3f}")
        
    def forward(self, x, labels):
        # print("ConvCVAEPL forward called")
        return self.model(x, labels)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        recon_batch, mu, logvar = self.model(x, labels)
        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        recon_batch, mu, logvar = self.model(x, labels)
        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, labels = batch
        recon_batch, mu, logvar = self.model(x, labels)
        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    