import layers
import torch
import torch.nn as nn

def coatten_sanity_check():
    bs = 1
    c_len = 3
    q_len = 5
    hid_size = 2
    att_layer = layers.CoAttention(hidden_size = 2 * hid_size)
    c_enc = torch.randn(bs, c_len, 2 * hid_size)
    q_enc = torch.randn(bs, q_len, 2* hid_size)
    c_mask = torch.randn(bs, c_len, 1) > 0
    q_mask = torch.randn(bs, 1, q_len) > 0
    [c_att, q_att] = att_layer(c_enc, q_enc, c_mask, q_mask)
    assert c_att.size() == torch.Size([1, 3, 4])
    assert q_att.size() == torch.Size([1, 5, 4])
    print("-"*80)
    print("Sanity Check Passed for Co-Attention layer")
    print("-"*80)

def main():
    coatten_sanity_check()

if __name__ == '__main__':
    main()