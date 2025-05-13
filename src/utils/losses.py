import torch

class L1Loss_MSEstages(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
        self.alpha = alpha

    def forward(self, output, gt, ):
        l1 = self.l1(output['pred'], gt['gt'])
        mse_stages = []
        for i in range(len(output['u_list'])):
            mse_stages.append(self.mse(output['u_list'][i], gt['gt']))
        mse_stages = torch.mean(torch.stack(mse_stages))
        loss = l1 + self.alpha * mse_stages
        return loss, {"l1": l1, "mse_stages": mse_stages}

    def components(self):
        return ["l1", "mse_stages"]

class L1(torch.nn.Module):
    def __init__(self, ):
        super(L1, self).__init__()
        self.L1 = torch.nn.L1Loss()

    def forward(self, output, gt):
        gt = gt['gt']
        pred = output['pred']
        l1 = self.L1(pred, gt)
        return l1, None