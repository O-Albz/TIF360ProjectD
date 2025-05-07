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
    def __init__(self, latent_size = 30, num_classes = 17, device = 'auto'): #for spectro 128,163
        super(ConvCVAE, self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.device = device

        # Encoder
        self.encoder_conv1 = nn.Conv2d(2 + num_classes, 32, kernel_size=4, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        # self.encoder_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.linear1 = nn.Linear(128 * 12 * 12, 500) # input is 513 x 862 --> TODO
        
        self.encoder_fc1 = nn.Linear(256 * 53 * 32, 100)
        
        self.encoder_fc_mu = nn.Linear(100, latent_size)
        self.encoder_fc_logvar = nn.Linear(100, latent_size)

        # Decoder
        # self.decoder_fc1 = nn.Linear(latent_size + num_classes, 500)
        # self.decoder_fc2 = nn.Linear(500, 128 * 12 * 12) # change to correct size TODO
        self.decoder_fc1 = nn.Linear(latent_size + num_classes, 100)
        self.decoder_fc2 = nn.Linear(100, 256 * 53 * 32) # change to correct size TODO
        # self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.decoder_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_conv4 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1, output_padding=(1,0))#, output_padding=(1,1)
        
        
    def encoder(self, x, labels):
        # class conditioning
        # y = torch.argmax(labels, dim=1).reshape((x.shape[0], 1, 1, 1))

        # One-hot encode labels
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float() # [batch_size, num_classes]
        y_maps = y_onehot.unsqueeze(2).unsqueeze(3) # [B, C, 1, 1]
        y_maps = y_maps.expand(-1, -1, x.shape[2], x.shape[3]).to(device) # [B, C, H, W]

        # print("y_maps shape", y_maps.shape)
        # print("x shape", x.shape)
        # sys.stdout.flush()
        
        x = torch.cat((x, y_maps), dim=1).to(device) # [B, C+num_classes, H, W]
        # print("Encoder \n Conv input shape:", x.shape)
        # sys.stdout.flush()
        x = F.relu(self.encoder_conv1(x))
        # print("Conv1 output shape:", x.shape)
        # sys.stdout.flush()
        x = F.relu(self.encoder_conv2(x))
        # print("Conv2 output shape:", x.shape)
        # sys.stdout.flush()
        x = F.relu(self.encoder_conv3(x))
        # print("Conv3 output shape:", x.shape)
        # sys.stdout.flush()
        x = F.relu(self.encoder_conv4(x))
        # print("Conv4 output shape:", x.shape)
        # sys.stdout.flush()
        # x = F.relu(self.encoder_conv5(x))
        x = x.view(x.size(0), -1)
        # print("output shape:", x.shape)
        # sys.stdout.flush()
        x = F.relu(self.encoder_fc1(x))

        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar
    

    def unFlatten(self, x):
        return x.reshape(x.shape[0], 256, 32, 53)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std
    
    def decoder(self, z):
        # print("Decoder \n linear input shape:", z.shape)
        # sys.stdout.flush()
        z = F.relu(self.decoder_fc1(z))
        z = F.relu(self.decoder_fc2(z))
        # print("linear shape:", z.shape)
        # sys.stdout.flush()
        xdot = self.unFlatten(z)
        # print("Flatt:", z.shape)
        # sys.stdout.flush()
        xdot = F.relu(self.decoder_conv1(xdot))
        # print("Conv1 output shape:", xdot.shape)
        # sys.stdout.flush()
        xdot = F.relu(self.decoder_conv2(xdot))
        # print("Conv2 output shape:", xdot.shape)
        # sys.stdout.flush()
        xdot = F.relu(self.decoder_conv3(xdot))
        # print("Conv3 output shape:", xdot.shape)
        # sys.stdout.flush()
        xdot = F.relu(self.decoder_conv4(xdot))
        # print("Conv4 output shape:", xdot.shape)
        # sys.stdout.flush()
        # z = F.relu(self.decoder_conv5(z))
        return xdot
    
    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        
        # class conditioning
        # print("z shape", z.shape)
        # print("label shape", labels.shape)
        # sys.stdout.flush()

        labels = F.one_hot(labels, num_classes=self.num_classes).float()
        z = torch.cat((z, labels), dim=1)

        # print("z shape", z.shape)
        # sys.stdout.flush()

        pred = self.decoder(z)

        return pred, mu, logvar        
    
    def loss_function(self, recon_x, x, mu, logvar, alpha = 1, beta = 1):
        BCE = F.mse_loss(recon_x, x, reduction='sum') # prop use sigmoid if BCE shold be used
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return alpha * BCE + beta * KLD
    
    def sample(self, z, labels):
        # y = torch.argmax(labels, dim=1).reshape((z.shape[0], 1, 1, 1))
        # y = torch.ones(z.shape).to(device) * y
        # z = torch.cat((z, y), dim=1)
        z = z.to(device)
        labels = F.one_hot(labels, num_classes=self.num_classes).float().to(device)
        z = torch.cat((z, labels), dim=1)

        return self.decoder(z)
        
        
class ConvCVAEPL(pl.LightningModule):
    # def __init__(self, latent_size = 128, num_classes = 163, device = 'auto'):
    def __init__(self, latent_size = 30, num_classes = 17, device = 'auto'):
        super(ConvCVAEPL, self).__init__()
        self.model = ConvCVAE(latent_size, num_classes, device)

    def on_train_epoch_start(self):
        # Slowly increase beta
        ramp_period = 20
        self.beta = float(1 / (1 + torch.exp(torch.tensor(-2 * (self.current_epoch - ramp_period)))))
        self.print(f"Beta = {self.beta:.3f}")
        
    def forward(self, x, labels):
        # print("ConvCVAEPL forward called")
        return self.model(x, labels)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch

        # # Turn labels into one-hot encoding
        # labels = torch.nn.functional.one_hot(labels, num_classes=self.model.num_classes).float()

        x = x.to(device)
        labels = labels.to(device)
        recon_batch, mu, logvar = self.model(x, labels)
        # recon_batch, mu, logvar = self(x, labels)

        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        recon_batch, mu, logvar = self.model(x, labels)
        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        recon_batch, mu, logvar = self.model(x, labels)
        loss = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    