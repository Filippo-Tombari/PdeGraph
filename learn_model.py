import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from dataset import create_dataset
from core_model import GNN
from plots import trajectorytogif

class Learner():
    ''' Class used for model training and rollout prediction'''
    def __init__(self,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #training parameters
        self.lr = args.lr
        self.milestones = args.milestones
        self.loss = args.loss
        self.noise_var = args.noise_var
        self.model_chk_path = args.model_chk_path
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.dt = 0.02

        self.train_data, self.valid_data, self.test_data = create_dataset(self.device)
        if args.train_model:
            self.net = GNN(args).to(self.device)
        else:
            self.net = torch.load('checkpoints/pretrained_net.pt', map_location = self.device)

        self.optimizer = Adam(self.net.parameters(),self.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)

    def train(self):
        ''' Trains and validates the model'''
        train_size = self.train_data["trajs"][0].shape[0]
        rollout_train_loss = []
        rollout_valid_loss = []

        print("Start Training")
        for epoch in range(self.epochs):
            rollout_train_loss.clear()
            rollout_valid_loss.clear()
            # training
            for sim in range(len(self.train_data['trajs'])):
                u = self.train_data['trajs'][sim]
                edge_index = self.train_data['edge_index'][sim]
                edge_weights = self.train_data['edge_weights'][sim].repeat(self.batch_size - 1, 1, 1)
                n_b_nodes = self.train_data['n_b_nodes'][sim]

                for batch in range(0, train_size - 1, self.batch_size):
                    u_batch = u[batch:batch + self.batch_size]
                    target = u_batch
                    # add gaussian noise
                    u_batch[:, n_b_nodes:, 0] += (self.noise_var) ** (0.5) * torch.randn_like(u_batch[:, n_b_nodes:, 0])

                    # forward pass
                    du_net = self.net(u_batch[:self.batch_size - 1], edge_index, edge_weights)  # (bs,nodes,1)
                    du = (target[1:, :, 0] - u_batch[:-1, :, 0]) / self.dt
                    train_loss = self.loss(du_net[:, :, 0], du, n_b_nodes, mode='train')
                    rollout_train_loss.append(train_loss.item())
                    # backpropagation
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()

            self.scheduler.step()

            # validation
            with torch.no_grad():
                for sim in range(len(self.valid_data['trajs'])):
                    u = self.valid_data['trajs'][sim]
                    edge_index = self.valid_data['edge_index'][sim]
                    edge_weights = self.valid_data['edge_weights'][sim].repeat(train_size - 1, 1, 1)
                    n_b_nodes = self.valid_data['n_b_nodes'][sim]

                    # forward pass
                    du_net = self.net(u[:train_size - 1], edge_index, edge_weights)
                    du = (u[1:, :, 0] - u[:-1, :, 0]) / self.dt
                    valid_loss = self.loss(du_net[:, :, 0], du, n_b_nodes, mode='train')
                    rollout_valid_loss.append(valid_loss.item())

            # print rollout number and MSE for training and validation set at each epoch
            mse_train = sum(rollout_train_loss) / len(rollout_train_loss)
            mse_valid = sum(rollout_valid_loss) / len(rollout_valid_loss)
            print(f"Epoch {epoch+1:1f}: MSE_train {mse_train :6.6f}, MSE_valid {mse_valid :6.6f}")

        print("End Training")
        print("Saving model")
        torch.save(self.net, self.model_chk_path)

    def forecast(self, save_plot = True):
        ''' Performs simulation rollout across all test simulations '''
        steps = self.test_data['trajs'][0].shape[0] - 1
        print("Start Testing")
        for sim in range(len(self.test_data['trajs'])):
            u = self.test_data['trajs'][sim]
            edge_index = self.test_data['edge_index'][sim]
            edge_attr = self.test_data['edge_attr'][sim]
            n_b_nodes = self.test_data['n_b_nodes'][sim]
            mesh = self.test_data['mesh'][sim]
            u_net = torch.zeros(u.shape).to(self.device)
            u_net[0] = u[0]
            u0 = u_net[[0]]

            for i in range(steps):
                du_net = self.net(u0, edge_index, edge_attr)
                u1 = u0 + self.dt*du_net
                u1[:,:n_b_nodes,0] = u[i+1,:n_b_nodes,0]
                u1[:,:,1:] = u[i+1,:,1:]
                u_net[i+1] = u1[0].detach()
                u0 = u1.detach()

            error = self.loss(u_net[:,:,0],u[:,:,0],n_b_nodes, 'test', mesh)
            print(f"Test simulation {sim+1:1f}: L2_rel_error {error :6.6f}")
            if save_plot:
                trajectorytogif(u_net, self.dt, name=f"images/test_sim_{sim}_pred", mesh=mesh)


        print("End Testing")
