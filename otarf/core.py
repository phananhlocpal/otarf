"""OTARF as described in the paper, plus its ablation variants.

The model: anchor-relative history,
observation mask/age/anchor validity, local base forecast, receiver/lead
conditioned query, support-gated attention with a learned null logit and an
additive signed correction.

Variants (all trained from scratch, same recipe):
  core               the paper model.
  no_anchor          history given as the filled standardized series Z (as the
                     baselines see it) instead of R = M (Z - b); the anchor value
                     b is removed from the output base and from all features.
                     Mask, age and anchor validity are kept, so the information
                     matches the indicator baselines.
  no_cross           correction fixed at zero; forecast = local base forecast.
  mean_pool          uniform average over the same valid source patches; no
                     query, key or null logit.
  state_blind        mask -> 1, age -> 0, anchor validity -> 1 at the input
                     (filled values, anchor and calendar kept), masked training kept.
  state_blind_no_aug state_blind and natural-view training only (handled by the trainer).
"""
import math
import torch
from torch import nn


def basis(h, k):
    """First k DCT-II basis vectors of length h (used for the query features)."""
    t = torch.arange(h, dtype=torch.float32).unsqueeze(1)
    f = torch.arange(k, dtype=torch.float32).unsqueeze(0)
    b = torch.cos(math.pi / h * (t + 0.5) * f) * math.sqrt(2.0 / h)
    b[:, 0] /= math.sqrt(2.0)
    return b

VARIANTS = ('core', 'no_anchor', 'no_cross', 'mean_pool', 'state_blind', 'state_blind_no_aug')


class OTARFCore(nn.Module):
    def __init__(self, channels, horizon, config, variant='core'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown OTARF-core variant '+str(variant))
        a = config['architecture']
        self.variant = variant
        self.c = channels; self.h = horizon; self.t = config['seq_len']; self.p = a['patch_length']
        self.lead = a['lead_group']; self.groups = horizon//self.lead
        if self.t % self.p or horizon % self.lead or a['basis_width'] > horizon or channels < 2:
            raise ValueError('Unsupported dimensions')
        self.np = self.t//self.p
        w = a['width']; e = a['identity_width']; d = a['attention_width']; v = a['value_width']; k = a['basis_width']
        self.register_buffer('basis', basis(horizon, k))
        self.local_id = nn.Embedding(channels, e); self.source_id = nn.Embedding(channels, e)
        self.receiver_id = nn.Embedding(channels, e); self.patch_id = nn.Embedding(self.np, e)
        self.group_id = nn.Embedding(self.groups, e)
        self.local_encoder = nn.Sequential(nn.Linear(4*self.t+7+e, w), nn.GELU(), nn.Linear(w, w), nn.GELU())
        self.local_head = nn.Linear(w, horizon); self.local_skip = nn.Linear(self.t, horizon, bias=False)
        self.patch_encoder = nn.Sequential(nn.Linear(4*self.p+2+2*e, w), nn.GELU(), nn.Linear(w, w), nn.GELU())
        self.query = nn.Linear(k+4+2*e, d); self.key = nn.Linear(w, d); self.value = nn.Linear(w, v)
        self.correction_head = nn.Linear(v, self.lead, bias=False)
        self.null_logit = nn.Parameter(torch.tensor(0.))
        self.correction_scale = nn.Parameter(torch.tensor(math.atanh(a['correction_scale_initial'])))

    @staticmethod
    def blind(mark, channels):
        ones = torch.ones_like(mark[:, :, :channels])
        # mark layout: mask c | age c | anchor validity c | anchor value c | calendar 4
        return torch.cat([ones, torch.zeros_like(ones), ones, mark[:, :, 3*channels:]], -1)

    def forward(self, x, mark, decoder=None, future_mark=None, return_details=False):
        B, T, C = x.shape
        if T != self.t or C != self.c:
            raise ValueError('Unexpected input shape')
        if self.variant.startswith('state_blind'):
            mark = self.blind(mark, C)
        mask, age, valid, anchors = (mark[:, :, i*C:(i+1)*C] for i in range(4))
        last_anchor = anchors[:, -1, :]; last_age = age[:, -1, :]; last_valid = valid[:, -1, :]
        if self.variant == 'no_anchor':
            rel = x.transpose(1, 2)
            base = torch.zeros_like(last_anchor)
        else:
            rel = torch.where(mask.bool(), x-last_anchor[:, None, :], torch.zeros_like(x)).transpose(1, 2)
            base = last_anchor
        hm = mask.transpose(1, 2); ha = torch.log1p(age).transpose(1, 2); hv = valid.transpose(1, 2)
        calendar = mark[:, -1, 4*C:4*C+4, None].transpose(1, 2).expand(-1, C, -1)
        feat = torch.cat([rel, hm, ha, hv, base.unsqueeze(-1), torch.log1p(last_age).unsqueeze(-1),
                          last_valid.unsqueeze(-1), calendar], -1)
        ids = torch.arange(C, device=x.device); lid = self.local_id(ids)[None].expand(B, -1, -1)
        local_state = self.local_encoder(torch.cat([feat, lid], -1))
        local = base.unsqueeze(-1)+self.local_skip(rel)+self.local_head(local_state)
        if self.variant == 'no_cross':
            self.correction_penalty = local.new_zeros(())
            self.telemetry = {'null_attention_mean': local.new_ones(()), 'correction_scale': torch.tanh(self.correction_scale).detach()}
            forecast = local.transpose(1, 2)
            return (forecast, forecast) if return_details else forecast

        def patches(a): return a.reshape(B, C, self.np, self.p)
        pm = patches(hm); counts = pm.sum(-1)
        patch_age = patches(age.transpose(1, 2))[:, :, :, -1]+torch.arange(self.np-1, -1, -1, device=x.device)[None, None, :]*self.p
        sid = self.source_id(ids)[None, :, None, :].expand(B, -1, self.np, -1)
        pid = self.patch_id(torch.arange(self.np, device=x.device))[None, None, :, :].expand(B, C, -1, -1)
        features = torch.cat([patches(rel), pm, patches(ha), patches(hv), counts.unsqueeze(-1)/self.p,
                              torch.log1p(patch_age).unsqueeze(-1), sid, pid], -1)
        source = self.patch_encoder(features); values = self.value(source)
        allowed = (counts > 0)[:, None, None, :, :] & ~torch.eye(C, dtype=torch.bool, device=x.device)[None, :, None, :, None]
        if self.variant == 'mean_pool':
            real = allowed.expand(B, C, self.groups, C, self.np).to(values.dtype)
            size = real.sum((3, 4), keepdim=True)
            real = real/size.clamp(min=1.)
            null_mean = (size == 0).to(values.dtype).mean()
        else:
            control = torch.cat([local@self.basis, base.unsqueeze(-1), torch.log1p(last_age).unsqueeze(-1),
                                 last_valid.unsqueeze(-1), hm.mean(-1, keepdim=True)], -1)
            rid = self.receiver_id(ids)[None, :, None, :].expand(B, -1, self.groups, -1)
            gid = self.group_id(torch.arange(self.groups, device=x.device))[None, None, :, :].expand(B, C, -1, -1)
            q = self.query(torch.cat([control[:, :, None, :].expand(-1, -1, self.groups, -1), rid, gid], -1))
            keys = self.key(source)
            logits = torch.einsum('bcgd,bjpd->bcgjp', q, keys)/math.sqrt(keys.shape[-1])
            logits = logits.masked_fill(~allowed, float('-inf')).reshape(B, C, self.groups, C*self.np)
            weights = torch.softmax(torch.cat([logits, self.null_logit.expand(B, C, self.groups, 1)], -1), -1)
            real = weights[..., :-1].reshape(B, C, self.groups, C, self.np)
            null_mean = weights[..., -1].mean()
        pooled = torch.einsum('bcgjp,bjpv->bcgv', real, values)
        correction = torch.tanh(self.correction_scale)*self.correction_head(pooled).reshape(B, C, self.h)
        self.correction_penalty = correction.square().mean()
        self.telemetry = {'null_attention_mean': null_mean.detach(), 'correction_scale': torch.tanh(self.correction_scale).detach()}
        forecast = (local+correction).transpose(1, 2)
        if return_details:
            return forecast, local.transpose(1, 2)
        return forecast
