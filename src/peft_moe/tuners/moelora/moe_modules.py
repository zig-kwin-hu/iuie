import torch
import torch.nn as nn
import torch.nn.functional as F
def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight
def _router_z_loss(logits):
    bsz, seq_len, _ = logits.size()
    log_z = torch.logsumexp(logits, dim=-1, keepdim=False)
    z_loss = log_z**2
    #z_loss = z_loss.sum() / (bsz*seq_len)
    return z_loss
class TopKGate(nn.Module):

    def __init__(self, input_size, expert_num, moe_topk: int = -1, loss_type=None, add_noise=False, regularized=True):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=-1)    # 第0维为batch size
        self.moe_topk = moe_topk
        self.expert_num = expert_num
        if loss_type == 'router_z':
            self.loss_function = _router_z_loss
        elif loss_type == 'no_loss':
            self.loss_function = lambda x: 0.
        elif loss_type is not None:
            raise ValueError(f'Unknown loss type: {loss_type}')
        else:
            self.loss_function = lambda x: 0.
        self.add_noise = add_noise
        self.regularized = regularized
            
    
    def forward(self, x):

        logits = self.GateL(x)
        
        if self.add_noise:
            logits = logits + torch.randn_like(logits)*((1/self.expert_num)**0.5)

        
        if self.moe_topk is not None and self.moe_topk > 0:
            #get the topk indices and generate a mask with only topk indices set to 1, then multiply with the gate output
            topk_logits, topk_indices = torch.topk(logits, min(self.moe_topk+1, self.expert_num), dim=-1)#batch_size, seq_len, topk+1
            topk_logits = topk_logits[:, :, :self.moe_topk]
            topk_indices = topk_indices[:, :, :self.moe_topk]
            zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
            if self.regularized:
                topk_scores = self.act(topk_logits.float()).to(logits.dtype)
            else:
                scores = self.act(logits.float()).to(logits.dtype)
                topk_scores = torch.gather(input=scores, dim=-1, index=topk_indices)
            scores_filtered = zeros.scatter(dim=-1, index=topk_indices, src=topk_scores)
        else:
            scores_filtered = self.act(logits.float()).to(logits.dtype)
        loss = None
        loss = self.loss_function(logits)
        return {'logits': logits, 'scores': scores_filtered, 'loss':loss}

class Expert(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.weight = self.mlp.weight
    

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y

class MOELinearA(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num, bias=False) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])

        #assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features# // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraA.append(Expert(self.in_features, self.r, bias))
        # cat the weights of the experts, shape is (out_features*expert_num, in_features)

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        #outputs = []
        #for i in range(self.expert_num):
            #outputs.append(self.loraA[i](x))
        self.loraA_cat = torch.cat([self.loraA[i].weight for i in range(self.expert_num)], dim=0)
        outputs = F.linear(x, self.loraA_cat)
        return outputs
    


class MOELinearB(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num, fan_in_fan_out, bias=False) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.fan_in_fan_out = fan_in_fan_out
        

        #assert self.in_features % self.expert_num == 0
        self.r = self.in_features# // self.expert_num
        self.loraB = nn.ModuleList([])        
        for _ in range(self.expert_num):
            self.loraB.append(Expert(self.r, self.out_features, bias))
        
        


    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        #outputs = []
        #for i in range(self.expert_num):
            #outputs.append(self.loraB[i](x[i]))
        self.loraB_block_diag = torch.block_diag(*[self.loraB[i].weight for i in range(self.expert_num)])
        outputs = F.linear(x, self.loraB_block_diag)#there is no need to consider the fan_in_fan_out, because the weight is generated locally
        shape = outputs.shape[:-1] + (self.expert_num, self.out_features)
        outputs = outputs.reshape(shape)
        
        return outputs