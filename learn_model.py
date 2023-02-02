import torch
import numpy as np
import random
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from dataset import create_dataset
from core_model import GNN
from plots import trajectorytogif

class Learner():
    ''' Class used for model training and rollout prediction'''
    def __init__(self,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Problem
        self.problem = args.example

        self.net = GNN(args).to(self.device)
        if not args.train_model:
            self.net.load_state_dict(torch.load('checkpoints/pretrained_net_' + f'{self.problem}', map_location = self.device))

        # Training parameters
        self.lr = args.lr
        self.milestones = args.milestones
        self.noise_var = args.noise_var
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.w1 = args.w1
        self.w2 = args.w2
        self.optimizer = Adam(self.net.parameters(), self.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
        self.dt = args.dt
        self.train_size = args.train_size

        # Dataset creation
        self.train_data, self.test_data = create_dataset(self.device, self.problem, self.train_size)


    def train(self):
        ''' Trains the model'''
        rollout_train_loss = []

        print("Start Training")
        for epoch in range(self.epochs):
            rollout_train_loss.clear()
            #shuffle data
            indices = list(range(len(self.train_data['trajs'])))
            random.shuffle(indices)
            # training
            for sim in indices:
                u = self.train_data['trajs'][sim]
                edge_index = self.train_data['edge_index'][sim]
                edge_weights = self.train_data['edge_weights'][sim].repeat(self.batch_size - 1, 1, 1)
                in_nodes = self.train_data['in_nodes'][sim]

                for batch in range(0, self.train_size - 1, self.batch_size):
                    u_batch = u[batch:batch + self.batch_size]
                    target = u_batch
                    # add gaussian noise
                    u_batch[:, in_nodes, 0] += (self.noise_var) ** (0.5) * torch.randn_like(u_batch[:, in_nodes, 0])

                    # forward pass
                    du_net = self.net(u_batch[:self.batch_size - 1], edge_index, edge_weights)  # (bs,nodes,1)
                    du = (target[1:, :, 0] - u_batch[:-1, :, 0]) / self.dt
                    train_loss_1 = ((du_net[:, :, 0] - du) ** 2)[:, in_nodes].mean()
                    u_net = u_batch[:-1, :, 0] + self.dt * du_net[:, :, 0]
                    if self.problem == 'Stokes':
                        u_net = u_net*(u_net>0) # The solution of Stokes problem
                                                # must be always non negative
                    train_loss_2 = ((u_net - target[1:, :, 0]) ** 2)[:, in_nodes].mean()
                    train_loss = self.w1*train_loss_1 + self.w2*train_loss_2
                    rollout_train_loss.append(train_loss.item())
                    # backpropagation
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()

            self.scheduler.step()

            # print rollout number and MSE for training set at each epoch
            mse_train = sum(rollout_train_loss) / len(rollout_train_loss)
            print(f"Epoch {epoch+1:1f}: MSE_train {mse_train :6.6f}")

        print("End Training")
        print("Saving model")
        idf = np.random.randint(100000)
        torch.save(self.net, f'checkpoints/chk_{idf}.pt')

    def forecast(self, save_plot = True):
        ''' Performs simulation rollout across all test simulations '''
        steps = self.test_data['trajs'][0].shape[0] - 1
        print("Start Testing")
        for sim in range(len(self.test_data['trajs'])):
            u = self.test_data['trajs'][sim]
            edge_index = self.test_data['edge_index'][sim]
            edge_attr = self.test_data['edge_weights'][sim]
            b_nodes = self.test_data['b_nodes'][sim]
            mesh = self.test_data['mesh'][sim]
            u_net = torch.zeros(u.shape).to(self.device)
            u_net[0] = u[0]
            u0 = u_net[[0]]

            for i in range(steps):
                du_net = self.net(u0, edge_index, edge_attr)
                u1 = u0 + self.dt*du_net
                if self.problem == 'Stokes':
                    u1 = u1*(u1>0)
                u1[:,b_nodes,0] = u[i+1,b_nodes,0]
                u1[:,:,1:] = u[i+1,:,1:]
                u_net[i+1] = u1[0].detach()
                u0 = u1.detach()

            error = (((u_net[:,:,0]-u[:,:,0])**2).sum(1)/(u[:,:,0]**2).sum(1)).mean()
            print(f"Test simulation {sim+1:1f}: L2_rel_error {error :6.6f}")
            if save_plot:
                trajectorytogif(u_net, self.dt, name=f"images/test_sim_{sim}_pred_"+f"{self.problem}", mesh=mesh)
                trajectorytogif(u, self.dt, name=f"images/test_sim_{sim}_true_"+f"{self.problem}", mesh=mesh)


        print("End Testing")
