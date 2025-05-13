import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
from torchmetrics.functional.image import spectral_angle_mapper as SAM
from torchmetrics.functional.image import error_relative_global_dimensionless_synthesis as ERGAS
from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR
from src.utils.non_ref_metrics import Q2N
from .non_ref_metrics import HQNR, QNR, D_s, D_lambda, D_f_lambda

metrics_dict = {
    'ERGAS': ERGAS,
    'PSNR': PSNR,
    'SSIM': SSIM,
    'SAM': SAM,
    'Q2N': Q2N,
}

non_ref_metrics_dict = {
    'HQNR': HQNR,
    'QNR': QNR,
    'Ds': D_s,
    'D_f_lambda': D_f_lambda,
    'D_lambda': D_lambda,
}

class MetricCalculator:
    def __init__(self,  metrics_list):
        self.metrics = {opt: metrics_dict[opt] for opt in metrics_list}
        self.dict = {k: [] for k in self.metrics.keys()}

    def update(self, preds, targets):
        for i in range(targets.size(0)):
            for k, v in self.metrics.items():
                match k:
                    case 'ERGAS':
                        self.dict[k].append(v(targets[i].unsqueeze(0), preds[i].unsqueeze(0)).cpu().detach().numpy())
                    case 'SAM':
                        aux = v(targets[i].unsqueeze(0), preds[i].unsqueeze(0)).cpu().detach().numpy()
                        self.dict[k].append(aux)
                    case'Q2N':
                        aux = v(targets[i].unsqueeze(0), preds[i].unsqueeze(0))
                        self.dict[k].append(aux)
                    case _:
                        self.dict[k].append(v(targets[i].unsqueeze(0), preds[i].unsqueeze(0), data_range=1).cpu().detach().numpy())

    def clean(self):
        self.dict = {k: [] for k in self.metrics.keys()}

    def get_statistics(self):
        mean_dict = {f"{k}": np.mean(v) for k, v in self.dict.items()}
        std_dict = {f"{k}": np.std(v) for k, v in self.dict.items()}
        return {"mean": mean_dict, "std": std_dict}

    def get_means(self):
        mean_dict = {f"{k}": np.mean(v) for k, v in self.dict.items()}
        return mean_dict

    def get_dict(self):
        return self.dict


class NonRefMetricCalculator:
    def __init__(self,  metrics_list):
        self.metrics = {opt: non_ref_metrics_dict[opt] for opt in metrics_list}
        self.dict = {k: [] for k in self.metrics.keys()}

    def update(self, pred, hs, pan):
        for i in range(pred.size(0)):
            print(pred.size(), hs.size(), pan.size())
            if 'HQNR' in self.metrics.keys() or 'D_f_lambda' in self.metrics.keys() or 'FQNR' in self.metrics.keys():
                D_f_lambda_value = self.metrics['D_f_lambda'](pred=pred[i].unsqueeze(0), ms=hs[i].unsqueeze(0))
            if 'QNR' in self.metrics.keys() or 'HQNR' in self.metrics.keys() or 'D_s' in self.metrics.keys():
                D_s_value = self.metrics['Ds'](pred=pred[i].unsqueeze(0), pan=pan[i].unsqueeze(0),ms=hs[i].unsqueeze(0))
            if 'QNR' in self.metrics.keys() or 'D_lambda' in self.metrics.keys():
                D_lambda_value = self.metrics['D_lambda'](pred=pred[i].unsqueeze(0), ms=hs[i].unsqueeze(0))
            for k, v in self.metrics.items():
                match k:
                    case 'HQNR':
                        self.dict[k].append(v(D_f_lambda_value, D_s_value).cpu().detach().numpy())
                    case 'QNR':
                        self.dict[k].append(v(D_lambda_value, D_s_value).cpu().detach().numpy())
                    case 'Ds':
                        self.dict[k].append(torch.tensor(D_s_value).cpu().detach().numpy())
                    case 'D_f_lambda':
                        self.dict[k].append(torch.tensor(D_f_lambda_value).cpu().detach().numpy())
                    case 'D_lambda':
                        self.dict[k].append(torch.tensor(D_lambda_value).cpu().detach().numpy())


    def clean(self):
        self.dict = {k: [] for k in self.metrics.keys()}

    def get_statistics(self):
        mean_dict = {f"{k}": np.mean(v) for k, v in self.dict.items()}
        std_dict = {f"{k}": np.std(v) for k, v in self.dict.items()}
        return {"mean": mean_dict, "std": std_dict}
    def get_means(self):
        mean_dict = {f"{k}": np.mean(v) for k, v in self.dict.items()}

        return mean_dict
    def get_dict(self):
        return self.dict