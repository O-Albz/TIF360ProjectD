import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
import librosa
import IPython.display as ipd
# from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class ConvCVAE(nn.Module):
    def __init__(self, latent_size = 128, num_classes = 17, device = 'auto', beta = 0.01): #for spectro 128,163
        super(ConvCVAE, self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.device = device
        self.beta = beta

        # Encoder
        self.encoder_conv1 = nn.Conv2d(2 + num_classes, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_batch_norm1 = nn.BatchNorm2d(32)

        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_batch_norm2 = nn.BatchNorm2d(64)

        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_batch_norm3 = nn.BatchNorm2d(128)

        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_batch_norm4 = nn.BatchNorm2d(256)

        self.encoder_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.encoder_fc_mu = nn.Linear(512 * 2 * 2, latent_size)
        self.encoder_fc_logvar = nn.Linear(512 * 2 * 2, latent_size)

        # Decoder
        self.decoder_fc = nn.Linear(latent_size + num_classes, 512 * 16 * 26)

        # make each latent_size + num_classes into a 16x26 image instead of above 

        # self.decoder_conv1 = nn.ConvTranspose2d(latent_size + num_classes, 256, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_batch_norm1 = nn.BatchNorm2d(256)
        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_batch_norm2 = nn.BatchNorm2d(128)
        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_batch_norm3 = nn.BatchNorm2d(64)
        self.decoder_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0,1))
        self.decoder_batch_norm4 = nn.BatchNorm2d(32)
        self.decoder_conv5 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1, output_padding=(1,0))#, output_padding=(1,1)
        
        # instead of unflattening, generate an image for the decoder all pixels in the channels are the same values
        
        self.dropout = nn.Dropout(0.3)
        self.dropout_conv = nn.Dropout2d(0.2)

    def encoder(self, x, labels):
        device = x.device
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        y_maps = y_onehot.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3]).to(device)
        x = torch.cat((x, y_maps), dim=1)

        x = F.leaky_relu(self.encoder_conv1(x))
        x = self.max_pool1(x)
        x = self.encoder_batch_norm1(x)
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.encoder_conv2(x))
        x = self.max_pool2(x)
        x = self.encoder_batch_norm2(x)
        x = F.leaky_relu(self.encoder_conv3(x))
        x = self.max_pool3(x)
        x = self.encoder_batch_norm3(x)
        x = self.dropout_conv(x)
        x = F.leaky_relu(self.encoder_conv4(x))
        x = self.max_pool4(x)
        x = self.encoder_batch_norm4(x)
        x = F.leaky_relu(self.encoder_conv5(x))

        x = self.adaptive_pool(x)
        x = x.view(x.shape[0], -1)

        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar

# patcg discriminwtor
# mse in the gan

# take out KLD by using CGAN instead, bottleneck sample normal distribution as usual.
# Discriminator will learn offset of mean. 
    def unFlatten(self, x):
        return x.reshape(x.shape[0], 512, 16, 26)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Automatically on correct device
        return mu + eps * std

    
    def decoder(self, z):
        xdot = F.leaky_relu(self.decoder_conv1(z))
        xdot = self.decoder_batch_norm1(xdot)
        xdot = self.dropout_conv(xdot)
        xdot = F.leaky_relu(self.decoder_conv2(xdot))
        xdot = self.decoder_batch_norm2(xdot)
        xdot = F.leaky_relu(self.decoder_conv3(xdot))
        xdot = self.decoder_batch_norm3(xdot)
        xdot = self.dropout_conv(xdot)
        xdot = F.leaky_relu(self.decoder_conv4(xdot))
        xdot = self.decoder_batch_norm4(xdot)
        xdot = self.decoder_conv5(xdot)
        return xdot
    
    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)

        labels = F.one_hot(labels, num_classes=self.num_classes).float().to(z.device)
        z = torch.cat((z, labels), dim=1)

        z = F.leaky_relu(self.decoder_fc(z))  # shape: [B, 512 * 16 * 26]
        z = z.view(-1, 512, 16, 26)     # shape: [B, 512, 16, 26]

        # z = z.unsqueeze(-1).unsqueeze(-1)
        # z = z.repeat(1, 1, 16, 26)

        pred = self.decoder(z)
        return pred, mu, logvar
        
    
    def loss_function(self, recon_x, x, mu, logvar, alpha = 1):
        BCE = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / x.size(0)
        return alpha * BCE + self.beta * KLD, BCE, KLD
    
    def sample(self, z, labels):
        device = z.device
        labels = labels.to(torch.long).to(device)
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(device)
        z = torch.cat((z, one_hot_labels), dim=1)

        z = F.leaky_relu(self.decoder_fc(z))  # shape: [B, 512 * 16 * 26]
        z = z.view(-1, 512, 16, 26)     # shape: [B, 512, 16, 26]

        return self.decoder(z)


class ConvCVAEPL(pl.LightningModule):
    def __init__(self, latent_size = 128, num_classes = 17, device = 'auto', learning_rate = 1e-3, beta = 0.01, ramp_beta = True, sample_every = 2):
        super(ConvCVAEPL, self).__init__()
        self.model = ConvCVAE(latent_size, num_classes, device, beta)
        self.learning_rate = learning_rate
        self.beta = beta
        self.sample_every = sample_every
        self.ramp_beta = ramp_beta

    def on_train_epoch_start(self):
        # Slowly increase beta
        if self.ramp_beta:
            # xshift = 7
            # yamp = 0.1
            # xamp = 0.4
            # max_beta = 2
            xshift = 11
            yamp = 0.1
            xamp = 0.25
            max_beta = 2
            self.beta = torch.minimum(yamp * torch.exp(torch.tensor(xamp * (self.current_epoch - xshift))), torch.tensor(max_beta))
            # ramp_period = 25
            # amplitude = 0.15
            # self.beta = float(1 / (1 + torch.exp(torch.tensor(-amplitude * (self.current_epoch - ramp_period)))))
        
        self.model.beta = self.beta
        # self.print(f"Beta = {self.beta:.5f}")
        self.log('beta', self.model.beta, prog_bar=True, on_epoch=True)
        
    def forward(self, x, labels):
        return self.model(x, labels)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        recon_batch, mu, logvar = self.model(x, labels)

        loss, recon, KLD = self.model.loss_function(recon_batch, x, mu, logvar)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_recon', recon, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_KLD', KLD, prog_bar=True, on_step=False, on_epoch=True)
        self.log("logvar_mean", logvar.mean(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        recon_batch, mu, logvar = self.model(x, labels)
        loss, _, _ = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, labels = batch
        recon_batch, mu, logvar = self.model(x, labels)
        loss, _, _ = self.model.loss_function(recon_batch, x, mu, logvar)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        train_recon = metrics.get("train_recon")
        train_KLD = metrics.get("train_KLD")
        val_loss = metrics.get("val_loss")
        beta = metrics.get("beta")

        if all(m is not None for m in [train_loss, train_recon, train_KLD, val_loss]):
            self.print(
                f"[Epoch {self.current_epoch}], Beta: {beta:.4f}, "
                f"Train Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KLD: {train_KLD:.4f} "
                f", Val Loss: {val_loss:.4f}"
            )

        if self.current_epoch % self.sample_every == 0:
            # sample a random latent vector for a random class
            label = torch.randint(0, self.model.num_classes, (1,)).to(self.device)
            z = torch.randn(1, self.model.latent_size).to(self.device)
            generated_sample = self.model.sample(z, label)
            
            D_real_mean = -0.00022800823861055075
            D_real_std = 3.1135954811248605
            D_imag_mean = -2.303626924950437e-08
            D_imag_std = 2.8981973344661305
            
            D_real = generated_sample[0][0].cpu().detach().numpy()
            D_real = (D_real * (D_real_std + 1e-8) * 16) + D_real_mean 
            D_imag = generated_sample[0][1].cpu().detach().numpy()
            D_imag = (D_imag * (D_imag_std + 1e-8) * 16) + D_imag_mean
            
            D = D_real + 1j * D_imag
            
            y = librosa.core.istft(D, hop_length=512, win_length=1024, window='hann')
            ipd.display(ipd.Audio(y, rate= 44100))