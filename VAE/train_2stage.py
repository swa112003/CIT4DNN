from torch.nn import functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import argparse
import matplotlib.pyplot as plt
from network import *
from VAE_utils import *
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import save_image
from torch.optim.lr_scheduler import *

def train(epoch, train_loader, stage):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        #lr = args.lr2 if args.lr_epochs2 <= 0 else args.lr2 * math.pow(args.lr_fac2, math.floor(float(epoch) / float(args.lr_epochs2)))
        
        data = data.to(device)
        optimizer.zero_grad()
        
        # Run VAE
        recon_batch, mu, logvar = vae(data, stage)
        # Compute loss
        rec, kl = vae.loss_function(recon_batch, data, mu, logvar, stage)
        
        total_loss = rec + args.beta*kl
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}\tKL: {:.6f}\tlog_sigma: {:f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                rec.item() / len(data),
                kl.item() / len(data),
                vae.log_sigma)) #.item() for learnable sigma
            
    train_loss /=  len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))
    summary_writer.add_scalar('train/elbo', train_loss, epoch)
    summary_writer.add_scalar('train/rec', rec.item() / len(data), epoch)
    summary_writer.add_scalar('train/kld', kl.item() / len(data), epoch)
    summary_writer.add_scalar('train/log_sigma', vae.log_sigma, epoch)


def test(epoch, test_loader, stage):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            recon_batch, mu, logvar = vae(data, stage)
            # Pass the second value from posthoc VAE
            rec, kl = vae.loss_function(recon_batch, data, mu, logvar, stage)
            test_loss += rec + kl
            if i == 0 and stage == 1:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, channels, width, height)[:n]])
                save_image(comparison.cpu(), 'vae_logs/{}/reconstruction_{}.png'.format(log_dir, str(epoch)), nrow=n)
                
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    summary_writer.add_scalar('test/elbo', test_loss, epoch)

def generate_stage2_dataset(train_loader):
    mean_stage1 = np.empty((0,args.latent_dim))
    log_var_stage1 = np.empty((0,args.latent_dim))
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            mean_stage_temp, log_var_stage_temp = vae.encoder(data)
            mean_stage_temp = mean_stage_temp.to(torch.device("cpu"))
            log_var_stage_temp = log_var_stage_temp.to(torch.device("cpu"))
            mean_stage1 = np.concatenate((mean_stage1,mean_stage_temp), axis=0)
            log_var_stage1 = np.concatenate((log_var_stage1,log_var_stage_temp), axis=0)
    print(f"Mean range Stage1 {np.amin(mean_stage1)} {np.amax(mean_stage1)}")
    print(f"logvar range Stage1 {np.amin(log_var_stage1)} {np.amax(log_var_stage1)}")
    
    tensor_mean_stage1 = torch.Tensor(mean_stage1)
    tensor_log_var_stage1 = torch.Tensor(log_var_stage1)
    stage2_dataset = TensorDataset(tensor_mean_stage1,tensor_log_var_stage1)
    stage2_dataloader = DataLoader(stage2_dataset, shuffle=True, batch_size=args.batch_size)
    print(len(stage2_dataloader), len(stage2_dataloader.dataset))
    return stage2_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "Udacity", "TaxiNet"], default="MNIST")
    parser.add_argument('--model', type=str, default='optimal_sigma_vae', metavar='N',
                    help='which model to use: mse_vae,  bce_vae, gaussian_vae, or sigma_vae or optimal_sigma_vae')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage2_epochs", type=int, default=200)
    parser.add_argument('--train_only_stage2', action='store_true')
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    torch.manual_seed(100)
    
    ## Logging
    log_dir, index_suffix = create_logdir("TwoStage", args.dataset, args.latent_dim, args.index)
    with open('vae_logs/{}/metadata.txt'.format(log_dir), 'w') as f:
        f.write(str(args))
        
    summary_writer = SummaryWriter(log_dir='vae_logs/' + log_dir, purge_step=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = eval("get_"+args.dataset+"_dataloader")(args.batch_size)
    test_loader = eval("get_"+args.dataset+"_dataloader")(args.batch_size, train=False)
    
    print(f"{args.dataset} dataset size is {len(train_loader)*args.batch_size}")
    
    data, _ = next(iter(train_loader))
    print(f"shape of {args.dataset} is {data.size()}")
    print(f"max and min of dataset tensor {torch.max(data)} and {torch.min(data)}")
    
    channels, width, height = get_dimensions(args.dataset)
    input_suffix = get_input_suffix(args.dataset)
    vae = TwoStageVAE(args.model, input_suffix, channels, args.latent_dim).to(device)
    
    if not torch.cuda.is_available():
        summary(vae, (channels, width, height))
        
    vae.train()

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    #scheduler = StepLR(optimizer, step_size=30)    
    scheduler = LinearLR(optimizer, total_iters=args.stage1_epochs)
    #training stage1
    if not args.train_only_stage2:
        start = time.time()
        print("Training Stage 1...")
        for epoch in range(1, args.stage1_epochs + 1):
            train(epoch, train_loader, 1)
            test(epoch, test_loader, 1)
            scheduler.step()
            if epoch%5==0 or epoch == args.stage1_epochs:
                with torch.no_grad():
                    sample = vae.sample(64, 1).cpu()
                    save_image(sample.view(64, channels, width, height),
                               'vae_logs/{}/stage1_sample_{}.png'.format(log_dir, str(epoch)))
                    summary_writer.file_writer.flush()
            #if epoch%30 == 0:
            #    torch.save(vae.state_dict(), 'vae_logs/{}/checkpoint_{}_stage1.pt'.format(log_dir, str(epoch)))
        torch.save(vae.state_dict(), f"./models/{args.dataset}_TwoStage_z{args.latent_dim}_stage1{index_suffix}.pth")
        print(f"Training time {(time.time() - start)/60} minutes")
    else:
        vae.load_state_dict(torch.load(f"./models/{args.dataset}_TwoStage_z{args.latent_dim}_stage1{index_suffix}.pth"))
        
    stage2_trainloader = generate_stage2_dataset(train_loader)
    stage2_testloader = generate_stage2_dataset(test_loader)
    
    args.beta = 1
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    #scheduler = StepLR(optimizer, step_size=60) 
    scheduler = LinearLR(optimizer, total_iters=args.stage2_epochs)
    start = time.time()
    print("Training Stage 2...")
    for epoch in range(1, args.stage2_epochs + 1):
        train(epoch, stage2_trainloader, 2)
        test(epoch, stage2_testloader, 2)
        scheduler.step()
        if epoch%5 == 0 or epoch == args.stage2_epochs:
            with torch.no_grad():
                sample = vae.sample(64, 2).cpu()
                
                save_image(sample.view(64, channels, width, height),
                           'vae_logs/{}/stage2_sample_{}.png'.format(log_dir, str(epoch)))
                summary_writer.file_writer.flush()
        #if epoch%50 == 0:
        #    torch.save(vae.state_dict(), 'vae_logs/{}/checkpoint_{}_stage2.pt'.format(log_dir, str(epoch)))
    torch.save(vae.state_dict(), f"./models/{args.dataset}_TwoStage_z{args.latent_dim}_stage2{index_suffix}.pth")
    print(f"Training time {(time.time() - start)/60} minutes")

    

    
