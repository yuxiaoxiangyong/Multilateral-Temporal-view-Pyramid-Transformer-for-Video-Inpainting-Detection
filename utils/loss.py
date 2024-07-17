import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=0.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()        
            #self.alpha = torch.tensor([alpha, 1-alpha]).cuda()    
            self.alpha = torch.tensor([1, 1]).cuda()      
            self.gamma = gamma 
            
    def forward(self, targets, inputs):
            #bce_loss = F.binary_cross_entropy(torch.sigmoid(inputs), targets)
            # 1 - aplha 代表困难样本的weight
            # gamma 代表缩放系数
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
            targets = targets.type(torch.long)        
            at = self.alpha.gather(0, targets.data.view(-1)).view(inputs.shape[0], -1) # gather  
            pt = torch.exp(-BCE_loss)         
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            #return bce_loss
            return F_loss.mean()
    

def softIoU(target, out, e=1e-6, recall=False):

    out = torch.sigmoid(out)
    if not recall:
        num = (out*target).sum(1,True).sum(-1,True)
        den = (out+target-out*target).sum(1,True).sum(-1,True) + e
        iou = num / den
    else:
        num = (out*target).sum(1,True).sum(-1,True)
        den = target.sum(1,True).sum(-1,True) + e
        iou = num / den        

    cost = (1 - iou)

    return cost.squeeze()


class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()

    def forward(self, y_true, y_pred, sw=None,recall=False):
        costs = softIoU(y_true,y_pred,recall).view(-1,1)
        if sw and (sw.data > 0).any():
            costs = torch.mean(torch.masked_select(costs,sw.byte()))
        else:
            costs = torch.mean(costs)

        return costs 