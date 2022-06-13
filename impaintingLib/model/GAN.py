from os.path import exists
from torchvision.datasets.folder import ImageFolder
import torch
import torch.nn as nn
import torchvision.models as models

def initGan():

    criterion = nn.BCELoss()
    real_label = 0.9
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    lr = 0.001
    beta1 = 0.5

    transform = transforms.Compose(
                  [transforms.Resize((120, 120)),
                   transforms.CenterCrop((64, 64)),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    inv_normalize = transforms.Normalize(
       mean= [-m/s for m, s in zip((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
       std= [1/s for s in (0.5, 0.5, 0.5)]
    )

    # -------------

    def get_super_resolution_loader(path, **kwargs):
        process = transforms.Compose(
          [transforms.Resize((120, 120)), transforms.CenterCrop((64, 64)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = ImageFolder(path, process)
        lengths = [12000, 1233]
        train_set, val_set = torch.utils.data.random_split(dataset, lengths)
        return DataLoader(train_set, **kwargs), DataLoader(val_set, **kwargs)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def update_discriminator(discriminator, optimizer, x, fake, real_label=1, fake_label=0):
        discriminator.zero_grad()
        b_size = x.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        # Forward pass real batch through D
        output = discriminator(x).view(-1)
        
        # Calculate loss on all-real batch
        loss = criterion(output, label)
        D_x = output.mean().item()
        label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        loss += criterion(output, label) 
        D_G_z1 = output.mean().item()
        loss.backward()
        optimizer.step()
        
        return loss, D_x, D_G_z1
    
    def update_generator(generator, discriminator, optimizer, x, fake, real_label=1):
        generator.zero_grad()
        b_size = fake.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = discriminator(fake).view(-1)
        loss = nn.BCELoss()(output, label) + torch.nn.L1Loss()(fake, x)
        loss.backward()
        D_G_z2 = output.mean().item()
        optimizer.step()
        
        return loss, D_G_z2
    
    def train_context_encoder(generator, discriminator, optimizerG, optimizerD, loader, epochs=10, real_label=0.9, fake_label=0, show_images=True):
        for epoch in range(epochs):
            t = tqdm(loader)
            for x, _ in t:
                x = x.to(device)
                
                ## Train with all-fake batch
                x_co = random_cutout(x, 5, 60) # IL CONNAIT PAS RANDOM CUTOUT, FAUT FAIRE UN IMPORT MAIS REPERTOIRE EST PLUS HAUT
                fake = generator(x_co)

                # discriminator update
                errD, D_x, D_G_z1 = update_discriminator(discriminator, optimizerD, x, fake, real_label, fake_label)
                errG, D_G_z2 = update_generator(generator, discriminator, optimizerG, x, fake, real_label)

                t.set_description(f'epoch:{epoch}/{epochs} \tLoss_D:{errD.item():.4f} \tLoss_G:{errG.item():.4f} \tD(x):{D_x:.4f} \tD(G(z)):{D_G_z1:.4f} / {D_G_z2:.4f}')

    # -------------
                
    generator = nn.Sequential(
                UNet(), 
                nn.Tanh()
            ).to(device)

    discriminator = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            ).to(device)

    discriminator.apply(weights_init)
    generator.apply(weights_init)
    
    # SI ON IMPORT PARENT POUR CUTOUT ON PEUT AUSSI IMPORT GET_FACES
    dataset = ImageFolder('data/lfw', transform)
    trainloader =  DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    train_context_encoder(generator, discriminator, optimizerG, optimizerD, trainloader, epochs=3, real_label=0.9, fake_label=0)
    
    return generator, discriminator

# -------------

gan_pretrained_path = "./modelSave/gan.pth"
criterion = nn.BCELoss()

def gan_loss(x,x_hat):
    
    # SI ON IMPORT PARENT POUR CUTOUT ON PEUT AUSSI IMPORT MODEL_SAVE ET MODEL_LOAD
    if exists(gan_pretrained_path):
        model = initGan() 
        model.load_state_dict(torch.load(gan_pretrained_path))
        model.eval()
    else : 
        print("/!\ Le Gan n'est pas pre-entrainé il faut d'abord le re-entrainer /!\")
        model,_ = train_gan()
        torch.save(model.state_dict(), gan_pretrained_path)
        
    return criterion(x_hat,model(x))