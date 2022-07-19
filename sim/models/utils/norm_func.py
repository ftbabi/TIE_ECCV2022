import torch


def mean_std_attn(query, receiver_emb, sender_emb, attn_mask, eps=1e-5):
    # emb: bs, num_head, N, head_dim
    dim = receiver_emb.shape[-1]

    mean_receiver = torch.mean(receiver_emb, dim=-1, keepdim=True)
    mean_sender = torch.mean(sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
    mean_rs = (mean_receiver + mean_sender)

    
    r_square = torch.sum(receiver_emb * receiver_emb, dim=-1, keepdim=True)
    s_square = torch.sum(sender_emb * sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
    rs = torch.matmul(receiver_emb, sender_emb.transpose(-1, -2))
    # var_rs = rs + r_square + s_square + mean_rs*mean_rs - (2/dim) * mean_rs*(mean_receiver + mean_sender)
    var_rs = (2*rs + r_square + s_square)/dim  - mean_rs*mean_rs + eps
    std_rs = torch.sqrt(var_rs)

    mean_rs = mean_rs * (~attn_mask)
    std_rs = std_rs.masked_fill(attn_mask, 1.0)

    # Original attn
    attn = torch.matmul(query, sender_emb.transpose(-1, -2)) + torch.sum(query * receiver_emb, dim=-1, keepdim=True)

    return mean_rs, std_rs, attn


