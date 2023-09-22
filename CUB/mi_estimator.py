from CUB.mi_estimators import CLUB, CLUBSample, KNIFE
from torch import nn

class mi_estimator(nn.Module):
    def __init__(self, args, mi_type, concept_dim, residual_dim):
        super(mi_estimator, self).__init__()
        self.mi_type = mi_type
        if self.mi_type == "clubsample":
            self.estimator = CLUBSample(args, zc_dim=concept_dim, zd_dim=residual_dim)
        elif self.mi_type == "club":
            self.estimator = CLUB(args, zc_dim=concept_dim, zd_dim=residual_dim)
        elif self.mi_type == "knife":
            self.estimator = KNIFE(args, zc_dim=concept_dim, zd_dim=residual_dim)

    def forward(self, x, y):
        for param in self.estimator.parameters():
            param.requires_grad = False
        estimate_mi, _, _ = self.estimator(x, y)
        return estimate_mi # return mutual information that only optimizes prior network

    def estimator_loss(self, x, y):
        for param in self.estimator.parameters():
            param.requires_grad = True
        d_x = x.detach()
        d_y = y.detach()
        return self.estimator.learning_loss(d_x, d_y) # return learning loss that optimizes the estimator

    def get_parameters(self):
        return self.estimator.parameters()