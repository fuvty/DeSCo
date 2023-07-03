"""
Use DIAMNet from https://github.com/HKUST-KnowComp/NeuralSubgraphCounting
A way to process the embeddings from GNNs
"""
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
import math

_INF = -1e30


class DIAMNet(nn.Module):
    def __init__(
        self,
        pattern_dim,
        graph_dim,
        hidden_dim,
        act_func="relu",
        recurrent_steps=1,
        num_heads=4,
        mem_len=4,
        mem_init="mean",
        dropout=0.0,
        dropatt=0.0,
    ):
        super(DIAMNet, self).__init__()

        self.pattern_dim = pattern_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.mem_len = mem_len
        self.mem_init = mem_init
        self.recurrent_steps = recurrent_steps

        self.act = map_activation_str_to_layer(act_func)
        self.drop = nn.Dropout(dropout)
        self.p_layer = nn.Linear(pattern_dim, hidden_dim)
        self.g_layer = nn.Linear(graph_dim, hidden_dim)
        if mem_init.endswith("attn"):
            self.m_layer = MultiHeadAttn(
                query_dim=hidden_dim,
                key_dim=graph_dim,
                value_dim=graph_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropatt=dropatt,
                act_func="softmax",
            )
        elif mem_init.endswith("lstm"):
            self.m_layer = nn.LSTM(graph_dim, hidden_dim, batch_first=True)
        else:
            self.m_layer = self.g_layer
        self.p_attn = GatedMultiHeadAttn(
            query_dim=hidden_dim,
            key_dim=pattern_dim,
            value_dim=pattern_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            pre_lnorm=True,
            dropatt=dropatt,
            act_func="softmax",
        )
        self.g_attn = GatedMultiHeadAttn(
            query_dim=hidden_dim,
            key_dim=graph_dim,
            value_dim=graph_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            pre_lnorm=True,
            dropatt=dropatt,
            act_func="softmax",
        )
        self.m_attn = GatedMultiHeadAttn(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            pre_lnorm=True,
            dropatt=dropatt,
            act_func="softmax",
        )

        self.pred_layer1 = nn.Linear(
            self.mem_len * self.hidden_dim + 4, self.hidden_dim
        )
        self.pred_layer2 = nn.Linear(self.hidden_dim + 4, 1)

        # init
        scale = 1 / (self.hidden_dim**0.5)
        for layer in [self.p_layer, self.g_layer, self.pred_layer1]:
            nn.init.normal_(layer.weight, 0.0, scale)
            nn.init.zeros_(layer.bias)
        for layer in [self.pred_layer2]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        if isinstance(self.m_layer, nn.LSTM):
            for layer_weights in self.m_layer._all_weights:
                for w in layer_weights:
                    if "weight" in w:
                        weight = getattr(self.m_layer, w)
                        nn.init.orthogonal_(weight)
                    elif "bias" in w:
                        bias = getattr(self.m_layer, w)
                        if bias is not None:
                            nn.init.zeros_(bias)
        elif isinstance(self.m_layer, nn.Linear):
            nn.init.normal_(layer.weight, 0.0, scale)
            nn.init.zeros_(layer.bias)

    def forward(self, pattern, pattern_len, graph, graph_len):
        bsz = pattern_len.size(0)
        p_len, g_len = pattern.size(1), graph.size(1)
        plf, glf = pattern_len.float(), graph_len.float()
        inv_plf, inv_glf = 1.0 / plf, 1.0 / glf
        p_mask = (
            batch_convert_len_to_mask(pattern_len)
            if p_len == pattern_len.max()
            else None
        )
        g_mask = (
            batch_convert_len_to_mask(graph_len) if g_len == graph_len.max() else None
        )

        p, g = pattern, graph
        if g_mask is not None:
            mem = list()
            mem_mask = list()
            for idx in gather_indices_by_lens(graph_len):
                if self.mem_init.endswith("attn"):
                    m, mk = init_mem(
                        g[idx, : graph_len[idx[0]]],
                        g_mask[idx, : graph_len[idx[0]]],
                        mem_len=self.mem_len,
                        mem_init=self.mem_init,
                        attn=self.m_layer,
                    )
                elif self.mem_init.endswith("lstm"):
                    m, mk = init_mem(
                        g[idx, : graph_len[idx[0]]],
                        g_mask[idx, : graph_len[idx[0]]],
                        mem_len=self.mem_len,
                        mem_init=self.mem_init,
                        lstm=self.m_layer,
                    )
                else:
                    m, mk = init_mem(
                        g[idx, : graph_len[idx[0]]],
                        g_mask[idx, : graph_len[idx[0]]],
                        mem_len=self.mem_len,
                        mem_init=self.mem_init,
                        post_proj=self.m_layer,
                    )
                mem.append(m)
                mem_mask.append(mk)
            mem = torch.cat(mem, dim=0)
            mem_mask = torch.cat(mem_mask, dim=0)
        else:
            if self.mem_init.endswith("attn"):
                mem, mem_mask = init_mem(
                    keyvalue,
                    None,
                    mem_len=self.mem_len,
                    mem_init=self.mem_init,
                    attn=self.m_layer,
                )
            elif self.mem_init.endswith("lstm"):
                mem, mem_mask = init_mem(
                    keyvalue,
                    None,
                    mem_len=self.mem_len,
                    mem_init=self.mem_init,
                    lstm=self.m_layer,
                )
            else:
                mem, mem_mask = init_mem(
                    keyvalue,
                    None,
                    mem_len=self.mem_len,
                    mem_init=self.mem_init,
                    post_proj=self.m_layer,
                )
        for i in range(self.recurrent_steps):
            mem = self.p_attn(mem, p, p, p_mask)
            mem = self.g_attn(mem, g, g, g_mask)

        mem = mem.view(bsz, -1)
        y = self.pred_layer1(torch.cat([mem, plf, glf, inv_plf, inv_glf], dim=1))
        y = self.act(y)
        y = self.pred_layer2(torch.cat([y, plf, glf, inv_plf, inv_glf], dim=1))

        return y

    def increase_input_size(self, new_pattern_dim, new_graph_dim):
        assert new_pattern_dim >= self.pattern_dim and new_graph_dim >= self.graph_dim
        if new_pattern_dim != self.pattern_dim:
            new_p_layer = extend_dimensions(
                self.p_layer, new_input_dim=new_pattern_dim, upper=False
            )
            del self.p_layer
            self.p_layer = new_p_layer
        if new_graph_dim != self.graph_dim:
            new_g_layer = extend_dimensions(
                self.g_layer, new_input_dim=new_graph_dim, upper=False
            )
            del self.g_layer
            self.g_layer = new_g_layer

        if self.mem_init.endswith("attn"):
            self.m_layer.increase_input_size(
                self.hidden_dim, new_graph_dim, new_graph_dim
            )
        else:
            new_m_layer = extend_dimensions(
                self.m_layer, new_input_dim=new_graph_dim, upper=False
            )
            del self.m_layer
            self.m_layer = new_m_layer
        self.p_attn.increase_input_size(
            self.hidden_dim, new_pattern_dim, new_pattern_dim
        )
        self.g_attn.increase_input_size(self.hidden_dim, new_graph_dim, new_graph_dim)

        self.pattern_dim = new_pattern_dim
        self.graph_dim = new_graph_dim


"""
const definations
"""
_act_map = {
    "none": lambda x: x,
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(1 / 5.5),
    "prelu": nn.PReLU(),
    "gelu": nn.GELU(),
}


def map_activation_str_to_layer(act_str):
    try:
        return _act_map[act_str]
    except:
        raise NotImplementedError(
            "Error: %s activation fuction is not supported now." % (act_str)
        )


def get_multi_head_attn_vec(
    head_q, head_k, head_v, attn_mask=None, act_layer=None, dropatt=None
):
    bsz, qlen, num_heads, head_dim = head_q.size()
    scale = 1 / (head_dim**0.5)

    # [bsz x qlen x klen x num_heads]
    attn_score = torch.einsum("bind,bjnd->bijn", (head_q, head_k))
    attn_score.mul_(scale)
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_score.masked_fill_((attn_mask == 0).unsqueeze(1).unsqueeze(-1), _INF)
        elif attn_mask.dim() == 3:
            attn_score.masked_fill_((attn_mask == 0).unsqueeze(-1), _INF)

    # [bsz x qlen x klen x num_heads]
    if act_layer is not None:
        attn_score = act_layer(attn_score)
    if dropatt is not None:
        attn_score = dropatt(attn_score)

    # [bsz x qlen x klen x num_heads] x [bsz x klen x num_heads x head_dim] -> [bsz x qlen x num_heads x head_dim]
    attn_vec = torch.einsum("bijn,bjnd->bind", (attn_score, head_v))
    attn_vec = attn_vec.contiguous().view(bsz, qlen, -1)

    return attn_vec


class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        hidden_dim,
        num_heads,
        dropatt=0.0,
        act_func="softmax",
        add_zero_attn=False,
        pre_lnorm=False,
        post_lnorm=False,
    ):
        super(MultiHeadAttn, self).__init__()
        assert hidden_dim % num_heads == 0
        assert act_func in ["softmax", "sigmoid"]

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropatt = nn.Dropout(dropatt)

        head_dim = hidden_dim // num_heads

        self.q_net = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_net = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v_net = nn.Linear(value_dim, hidden_dim, bias=False)
        self.o_net = nn.Linear(hidden_dim, query_dim, bias=False)

        self.act = map_activation_str_to_layer(act_func)
        self.add_zero_attn = add_zero_attn
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm

        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
        if post_lnorm:
            self.o_layer_norm = nn.LayerNorm(query_dim)

        # init
        scale = 1 / (head_dim**0.5)
        for m in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.normal_(m.weight, 0.0, scale)

    def forward(self, query, key, value, attn_mask=None):
        ##### multihead attention
        # [bsz x hlen x num_heads x head_dim]
        bsz = query.size(0)

        if self.add_zero_attn:
            key = torch.cat(
                [
                    key,
                    torch.zeros(
                        (bsz, 1) + key.size()[2:], dtype=key.dtype, device=key.device
                    ),
                ],
                dim=1,
            )
            value = torch.cat(
                [
                    value,
                    torch.zeros(
                        (bsz, 1) + value.size()[2:],
                        dtype=value.dtype,
                        device=value.device,
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [
                        attn_mask,
                        torch.ones(
                            (bsz, 1), dtype=attn_mask.dtype, device=attn_mask.device
                        ),
                    ],
                    dim=1,
                )

        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)

        if self.pre_lnorm:
            ##### layer normalization
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)

        # linear projection
        head_q = self.q_net(query).view(
            bsz, qlen, self.num_heads, self.hidden_dim // self.num_heads
        )
        head_k = self.k_net(key).view(
            bsz, klen, self.num_heads, self.hidden_dim // self.num_heads
        )
        head_v = self.v_net(value).view(
            bsz, vlen, self.num_heads, self.hidden_dim // self.num_heads
        )

        # multi head attention
        attn_vec = get_multi_head_attn_vec(
            head_q=head_q,
            head_k=head_k,
            head_v=head_v,
            attn_mask=attn_mask,
            act_layer=self.act,
            dropatt=self.dropatt,
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)

        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)

        return attn_out

    def increase_input_size(self, new_query_dim, new_key_dim, new_value_dim):
        assert (
            new_query_dim >= self.query_dim
            and new_key_dim >= self.key_dim
            and new_value_dim >= new_value_dim
        )

        if new_query_dim != self.query_dim:
            new_q_net = extend_dimensions(
                self.q_net, new_input_dim=new_query_dim, upper=False
            )
            del self.q_net
            self.q_net = new_q_net
            if self.pre_lnorm:
                new_q_layer_norm = extend_dimensions(
                    self.q_layer_norm, new_input_dim=new_query_dim, upper=False
                )
                del self.q_layer_norm
                self.q_layer_norm = new_q_layer_norm

            new_o_net = extend_dimensions(
                self.o_net, new_output_dim=new_query_dim, upper=False
            )
            del self.o_net
            self.o_net = new_o_net
            if self.post_lnorm:
                new_o_layer_norm = extend_dimensions(
                    self.o_layer_norm, new_input_dim=new_query_dim, upper=False
                )
                del self.o_layer_norm
                self.o_layer_norm = new_o_layer_norm
        if new_key_dim != self.key_dim:
            new_k_net = extend_dimensions(
                self.k_net, new_input_dim=new_key_dim, upper=False
            )
            del self.k_net
            self.k_net = new_k_net
            if self.pre_lnorm:
                new_k_layer_norm = extend_dimensions(
                    self.k_layer_norm, new_input_dim=new_key_dim, upper=False
                )
                del self.k_layer_norm
                self.k_layer_norm = new_k_layer_norm
        if new_value_dim != self.value_dim:
            new_v_net = extend_dimensions(
                self.v_net, new_input_dim=new_value_dim, upper=False
            )
            del self.v_net
            self.v_net = new_v_net
            if self.pre_lnorm:
                new_v_layer_norm = extend_dimensions(
                    self.v_layer_norm, new_input_dim=new_value_dim, upper=False
                )
                del self.v_layer_norm
                self.v_layer_norm = new_v_layer_norm
        self.query_dim = new_query_dim
        self.key_dim = new_key_dim
        self.value_dim = new_value_dim


class GatedMultiHeadAttn(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        hidden_dim,
        num_heads,
        dropatt=0.0,
        act_func="softmax",
        add_zero_attn=False,
        pre_lnorm=False,
        post_lnorm=False,
    ):
        super(GatedMultiHeadAttn, self).__init__()
        assert hidden_dim % num_heads == 0

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropatt = nn.Dropout(dropatt)

        head_dim = hidden_dim // num_heads

        self.q_net = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_net = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v_net = nn.Linear(value_dim, hidden_dim, bias=False)
        self.o_net = nn.Linear(hidden_dim, query_dim, bias=False)
        self.g_net = nn.Linear(2 * query_dim, query_dim, bias=True)

        self.act = map_activation_str_to_layer(act_func)
        self.add_zero_attn = add_zero_attn
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm

        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
        if post_lnorm:
            self.o_layer_norm = nn.LayerNorm(query_dim)

        # init
        scale = 1 / (head_dim**0.5)
        for m in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.normal_(m.weight, 0.0, scale)
        # when new data comes, it prefers to output 1 so that the gate is 1
        nn.init.normal_(self.g_net.weight, 0.0, scale)
        nn.init.ones_(self.g_net.bias)

    def forward(self, query, key, value, attn_mask=None):
        ##### multihead attention
        # [bsz x hlen x num_heads x head_dim]
        bsz = query.size(0)

        if self.add_zero_attn:
            key = torch.cat(
                [
                    key,
                    torch.zeros(
                        (bsz, 1) + key.size()[2:], dtype=key.dtype, device=key.device
                    ),
                ],
                dim=1,
            )
            value = torch.cat(
                [
                    value,
                    torch.zeros(
                        (bsz, 1) + value.size()[2:],
                        dtype=value.dtype,
                        device=value.device,
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [
                        attn_mask,
                        torch.ones(
                            (bsz, 1), dtype=attn_mask.dtype, device=attn_mask.device
                        ),
                    ],
                    dim=1,
                )

        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)

        if self.pre_lnorm:
            ##### layer normalization
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)

        # linear projection
        head_q = self.q_net(query).view(
            bsz, qlen, self.num_heads, self.hidden_dim // self.num_heads
        )
        head_k = self.k_net(key).view(
            bsz, klen, self.num_heads, self.hidden_dim // self.num_heads
        )
        head_v = self.v_net(value).view(
            bsz, vlen, self.num_heads, self.hidden_dim // self.num_heads
        )

        # multi head attention
        attn_vec = get_multi_head_attn_vec(
            head_q=head_q,
            head_k=head_k,
            head_v=head_v,
            attn_mask=attn_mask,
            act_layer=self.act,
            dropatt=self.dropatt,
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)

        ##### gate
        gate = F.sigmoid(self.g_net(torch.cat([query, attn_out], dim=2)))
        attn_out = gate * query + (1 - gate) * attn_out

        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)

        return attn_out

    def increase_input_size(self, new_query_dim, new_key_dim, new_value_dim):
        assert (
            new_query_dim >= self.query_dim
            and new_key_dim >= self.key_dim
            and new_value_dim >= new_value_dim
        )

        if new_query_dim != self.query_dim:
            new_q_net = extend_dimensions(
                self.q_net, new_input_dim=new_query_dim, upper=False
            )
            del self.q_net
            self.q_net = new_q_net
            if self.pre_lnorm:
                new_q_layer_norm = extend_dimensions(
                    self.q_layer_norm, new_input_dim=new_query_dim, upper=False
                )
                del self.q_layer_norm
                self.q_layer_norm = new_q_layer_norm

            new_o_net = extend_dimensions(
                self.o_net, new_output_dim=new_query_dim, upper=False
            )
            del self.o_net
            self.o_net = new_o_net
            if self.post_lnorm:
                new_o_layer_norm = extend_dimensions(
                    self.o_layer_norm, new_input_dim=new_query_dim, upper=False
                )
                del self.o_layer_norm
                self.o_layer_norm = new_o_layer_norm

            new_g_net = nn.Linear(2 * new_query_dim, new_query_dim)
            with torch.no_grad():
                nn.init.zeros_(new_g_net.weight)
                nn.init.zeros_(new_g_net.bias)
                new_g_net.weight[
                    -self.query_dim :, new_query_dim - self.query_dim : new_query_dim
                ].data.copy_(self.g_net.weight[:, : self.query_dim])
                new_g_net.weight[-self.query_dim :, -self.query_dim :].data.copy_(
                    self.g_net.weight[:, -self.query_dim :]
                )
                new_g_net.bias[
                    new_query_dim - self.query_dim : new_query_dim
                ].data.copy_(self.g_net.bias[: self.query_dim])
                new_g_net.bias[-self.query_dim :].data.copy_(
                    self.g_net.bias[-self.query_dim :]
                )
                del self.g_net
                self.g_net = new_g_net

        if new_key_dim != self.key_dim:
            new_k_net = extend_dimensions(
                self.k_net, new_input_dim=new_key_dim, upper=False
            )
            del self.k_net
            self.k_net = new_k_net
            if self.pre_lnorm:
                new_k_layer_norm = extend_dimensions(
                    self.k_layer_norm, new_input_dim=new_key_dim, upper=False
                )
                del self.k_layer_norm
                self.k_layer_norm = new_k_layer_norm
        if new_value_dim != self.value_dim:
            new_v_net = extend_dimensions(
                self.v_net, new_input_dim=new_value_dim, upper=False
            )
            del self.v_net
            self.v_net = new_v_net
            if self.pre_lnorm:
                new_v_layer_norm = extend_dimensions(
                    self.v_layer_norm, new_input_dim=new_value_dim, upper=False
                )
                del self.v_layer_norm
                self.v_layer_norm = new_v_layer_norm
        self.query_dim = new_query_dim
        self.key_dim = new_key_dim
        self.value_dim = new_value_dim


def extend_dimensions(old_layer, new_input_dim=-1, new_output_dim=-1, upper=False):
    if isinstance(old_layer, nn.Linear):
        old_output_dim, old_input_dim = old_layer.weight.size()
        if new_input_dim == -1:
            new_input_dim = old_input_dim
        if new_output_dim == -1:
            new_output_dim = old_output_dim
        assert new_input_dim >= old_input_dim and new_output_dim >= old_output_dim

        if new_input_dim != old_input_dim or new_output_dim != old_output_dim:
            use_bias = old_layer.bias is not None
            new_layer = nn.Linear(new_input_dim, new_output_dim, bias=use_bias)
            with torch.no_grad():
                nn.init.zeros_(new_layer.weight)
                if upper:
                    new_layer.weight[:old_output_dim, :old_input_dim].data.copy_(
                        old_layer.weight
                    )
                else:
                    new_layer.weight[-old_output_dim:, -old_input_dim:].data.copy_(
                        old_layer.weight
                    )
                if use_bias:
                    nn.init.zeros_(new_layer.bias)
                    if upper:
                        new_layer.bias[:old_output_dim].data.copy_(old_layer.bias)
                    else:
                        new_layer.bias[-old_output_dim:].data.copy_(old_layer.bias)
        else:
            new_layer = old_layer
    elif isinstance(old_layer, nn.LayerNorm):
        old_input_dim = old_layer.normalized_shape
        if len(old_input_dim) != 1:
            raise NotImplementedError
        old_input_dim = old_input_dim[0]
        assert new_input_dim >= old_input_dim
        if new_input_dim != old_input_dim and old_layer.elementwise_affine:
            new_layer = nn.LayerNorm(new_input_dim, elementwise_affine=True)
            with torch.no_grad():
                nn.init.ones_(new_layer.weight)
                nn.init.zeros_(new_layer.bias)
                if upper:
                    new_layer.weight[:old_input_dim].data.copy_(old_layer.weight)
                    new_layer.bias[:old_input_dim].data.copy_(old_layer.bias)
                else:
                    new_layer.weight[-old_input_dim:].data.copy_(old_layer.weight)
                    new_layer.bias[-old_input_dim:].data.copy_(old_layer.bias)
        else:
            new_layer = old_layer
    elif isinstance(old_layer, nn.LSTM):
        old_input_dim, old_output_dim = old_layer.input_size, old_layer.hidden_size
        if new_input_dim == -1:
            new_input_dim = old_input_dim
        if new_output_dim == -1:
            new_output_dim = old_output_dim
        assert new_input_dim >= old_input_dim and new_output_dim >= old_output_dim

        if new_input_dim != old_input_dim or new_output_dim != old_output_dim:
            new_layer = nn.LSTM(
                new_input_dim,
                new_output_dim,
                num_layers=old_layer.num_layers,
                bidirectional=old_layer.bidirectional,
                batch_first=old_layer.batch_first,
                bias=old_layer.bias,
            )
            for layer_weights in new_layer._all_weights:
                for w in layer_weights:
                    with torch.no_grad():
                        if "weight" in w:
                            new_weight = getattr(new_layer, w)
                            old_weight = getattr(old_layer, w)
                            nn.init.zeros_(new_weight)
                            if upper:
                                new_weight[
                                    : old_weight.shape[0], : old_weight.shape[1]
                                ].data.copy_(old_weight)
                            else:
                                new_weight[
                                    -old_weight.shape[0] :, -old_weight.shape[1] :
                                ].data.copy_(old_weight)
                        if "bias" in w:
                            new_bias = getattr(new_layer, w)
                            old_bias = getattr(old_layer, w)
                            if new_bias is not None:
                                nn.init.zeros_(new_bias)
                                if upper:
                                    new_bias[: old_bias.shape[0]].data.copy_(old_bias)
                                else:
                                    new_bias[-old_bias.shape[0] :].data.copy_(old_bias)
    return new_layer


def batch_convert_len_to_mask(batch_lens, max_seq_len=-1):
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    mask = torch.ones(
        (len(batch_lens), max_seq_len),
        dtype=torch.uint8,
        device=batch_lens[0].device,
        requires_grad=False,
    )
    for i, l in enumerate(batch_lens):
        mask[i, l:].fill_(0)
    return mask


def gather_indices_by_lens(lens):
    result = list()
    i, j = 0, 1
    max_j = len(lens)
    indices = np.arange(0, max_j)
    while j < max_j:
        if lens[i] != lens[j]:
            result.append(indices[i:j])
            i = j
        j += 1
    if i != j:
        result.append(indices[i:j])
    return result


def init_mem(x, x_mask=None, mem_len=4, mem_init="mean", **kw):
    assert mem_init in [
        "mean",
        "sum",
        "max",
        "attn",
        "lstm",
        "circular_mean",
        "circular_sum",
        "circular_max",
        "circular_attn",
        "circular_lstm",
    ]
    pre_proj = kw.get("pre_proj", None)
    post_proj = kw.get("post_proj", None)
    if pre_proj:
        x = pre_proj(x)

    bsz, seq_len, hidden_dim = x.size()
    if seq_len < mem_len:
        mem = torch.cat(
            [
                x,
                torch.zeros(
                    (bsz, mem_len - seq_len, hidden_dim), device=x.device, dtype=x.dtype
                ),
            ],
            dim=1,
        )
        if x_mask is not None:
            mem_mask = torch.cat(
                [
                    x_mask,
                    torch.zeros(
                        (bsz, mem_len - seq_len),
                        device=x_mask.device,
                        dtype=x_mask.dtype,
                    ),
                ],
                dim=1,
            )
        else:
            mem_mask = None
    elif seq_len == mem_len:
        mem, mem_mask = x, x_mask
    else:
        if mem_init.startswith("circular"):
            pad_len = math.ceil((seq_len + 1) / 2) - 1
            x = F.pad(x.transpose(1, 2), pad=(0, pad_len), mode="circular").transpose(
                1, 2
            )
            if x_mask is not None:
                x_mask = F.pad(
                    x_mask.unsqueeze(1), pad=(0, pad_len), mode="circular"
                ).squeeze(1)
            seq_len += pad_len
        stride = seq_len // mem_len
        kernel_size = seq_len - (mem_len - 1) * stride
        if mem_init.endswith("mean"):
            mem = F.avg_pool1d(
                x.transpose(1, 2), kernel_size=kernel_size, stride=stride
            ).transpose(1, 2)
        elif mem_init.endswith("max"):
            mem = F.max_pool1d(
                x.transpose(1, 2), kernel_size=kernel_size, stride=stride
            ).transpose(1, 2)
        elif mem_init.endswith("sum"):
            mem = (
                F.avg_pool1d(
                    x.transpose(1, 2), kernel_size=kernel_size, stride=stride
                ).transpose(1, 2)
                * kernel_size
            )
        elif mem_init.endswith("attn"):
            # split and self attention
            mem = list()
            attn = kw.get("attn", None)
            hidden_dim = attn.query_dim
            h = torch.ones(
                (bsz, 1, hidden_dim),
                dtype=x.dtype,
                device=x.device,
                requires_grad=False,
            ).mul_(1 / (hidden_dim**0.5))
            for i in range(0, seq_len - kernel_size + 1, stride):
                j = i + kernel_size
                m = x[:, i:j]
                mk = x_mask[:, i:j] if x_mask is not None else None
                if attn:
                    h = attn(h, m, m, attn_mask=mk)
                else:
                    m = m.unsqueeze(2)
                    h = get_multi_head_attn_vec(
                        h, m, m, attn_mask=mk, act_layer=nn.Softmax(dim=-1)
                    )
                mem.append(h)
            mem = torch.cat(mem, dim=1)
        elif mem_init.endswith("lstm"):
            mem = list()
            lstm = kw["lstm"]
            hx = None
            for i in range(0, seq_len - kernel_size + 1, stride):
                j = i + kernel_size
                m = x[:, i:j]
                _, hx = lstm(m, hx)
                mem.append(hx[0].view(bsz, 1, -1))
            mem = torch.cat(mem, dim=1)
        if x_mask is not None:
            mem_mask = (
                F.max_pool1d(
                    x_mask.float().unsqueeze(1), kernel_size=kernel_size, stride=stride
                )
                .squeeze(1)
                .byte()
            )
        else:
            mem_mask = None
    if post_proj:
        mem = post_proj(mem)
    return mem, mem_mask


def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
    bsz = graph_sizes.size(0)
    dim, dtype, device = (
        batched_graph_feats.size(-1),
        batched_graph_feats.dtype,
        batched_graph_feats.device,
    )

    min_size, max_size = graph_sizes.min(), graph_sizes.max()
    mask = torch.ones(
        (bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False
    )

    if min_size == max_size:
        return batched_graph_feats.view(bsz, max_size, -1), mask
    else:
        graph_sizes_list = graph_sizes.view(-1).tolist()
        unbatched_graph_feats = list(
            torch.split(batched_graph_feats, graph_sizes_list, dim=0)
        )
        for i, l in enumerate(graph_sizes_list):
            if l == max_size:
                continue
            elif l > max_size:
                unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
            else:
                mask[i, l:].fill_(0)
                zeros = torch.zeros(
                    (max_size - l, dim), dtype=dtype, device=device, requires_grad=False
                )
                unbatched_graph_feats[i] = torch.cat(
                    [unbatched_graph_feats[i], zeros], dim=0
                )
        return torch.stack(unbatched_graph_feats, dim=0), mask
