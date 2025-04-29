import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class ConvCVAE(nn.Module):
    def __init__(self, latent_size = 128, num_classes = 163, device = 'auto'):
        super(ConvCVAE, self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.device = device

        # Encoder
        self.encoder_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.encoder_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(128 * 12 * 12, 500) # input is 513 x 862 --> TODO
        
        self.encoder_fc_mu = nn.Linear(500, latent_size)
        self.encoder_fc_logvar = nn.Linear(500, latent_size)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_size + num_classes, 500)
        self.decoder_fc2 = nn.Linear(500, 128 * 12 * 12) # change to correct size TODO
        self.encoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.encoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.decoder_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.decoder_conv5 = nn.ConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1)
        
        def encoder(self, x, labels):
            y = torch.argmax(labels, dim=1).reshape((x.shape[0], 1, 1, 1))
            y = torch.ones(x.shape).to(device) * y
            x = torch.cat((x, y), dim=1)
            x = F.relu(self.encoder_conv1(x))
            x = F.relu(self.encoder_conv2(x))
            x = F.relu(self.encoder_conv3(x))
            x = F.relu(self.encoder_conv4(x))
            x = F.relu(self.encoder_conv5(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.linear1(x))
            mu = self.encoder_fc_mu(x)
            logvar = self.encoder_fc_logvar(x)
            return mu, logvar
        
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(device)
            return mu + eps * std
        
        def sample(self, z, labels):
            y = torch.argmax(labels, dim=1).reshape((z.shape[0], 1, 1, 1))
            y = torch.ones(z.shape).to(device) * y
            z = torch.cat((z, y), dim=1)
            return self.decoder(z, labels)
        
        def decoder(self, z, labels):
            z = F.relu(self.decoder_fc1(z))
            z = F.relu(self.decoder_fc2(z))
            z = self.unFlatten(z)
            z = F.relu(self.decoder_conv1(z))
            z = F.relu(self.decoder_conv2(z))
            z = F.relu(self.decoder_conv3(z))
            z = F.relu(self.decoder_conv4(z))
            z = F.relu(self.decoder_conv5(z))
            return z
        
        def unFlatten(self, x):
            return x.reshape(x.shape[0], 128, 8, 8)
        
        def forward(self, x, labels):
            mu, logvar = self.encoder(x, labels)
            z = self.reparameterize(mu, logvar)
            
            # class conditioning
            z = torch.cat((z, labels.float()), dim=1)
            pred = self.decoder(z, labels)
            return pred, mu, logvar
        
        def loss_function(self, recon_x, x, mu, logvar, alpha = 1, beta = 1):
            BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')     
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return alpha * BCE + beta * KLD
        
        
class ConvCVAEPL(pl.LightningModule):
    def __init__(self, latent_size = 128, num_classes = 163, device = 'auto'):
        super(ConvCVAEPL, self).__init__()
        self.model = ConvCVAE(latent_size, num_classes, device)
        
    def forward(self, x, labels):
        return self.model(x, labels)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        recon_batch, mu, logvar = self.model(x, labels)
        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        recon_batch, mu, logvar = self.model(x, labels)
        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        recon_batch, mu, logvar = self.model(x, labels)
        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        
        