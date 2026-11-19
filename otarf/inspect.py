"""Inference-only inspection of a trained OTARF-core (variant 'core'); no training, weights unchanged.

forward_inspect(x, mark, groups, exclude) repeats OTARFCore.forward for the 'core' variant and additionally
  * returns the attention mass each receiver puts on every source group, the null weight, and the number of
    valid candidate segments per group (for the uniform reference);
  * optionally removes the sources of chosen groups from the valid source set (exclude[receiver, source]).
With exclude=None the forecast must equal OTARFCore.forward; the runner checks this on every batch.
"""
import math
import torch
from .core import OTARFCore


class OTARFCoreInspect(OTARFCore):
    def __init__(self, channels, horizon, config):
        super().__init__(channels, horizon, config, 'core')

    def forward_inspect(self, x, mark, groups, exclude=None):
        """groups: float one-hot [C_receiver, C_source, K] (zero row for the receiver itself);
        exclude: bool [C_receiver, C_source] or None."""
        B, T, C = x.shape
        mask, age, valid, anchors = (mark[:, :, i*C:(i+1)*C] for i in range(4))
        last_anchor = anchors[:, -1, :]; last_age = age[:, -1, :]; last_valid = valid[:, -1, :]
        rel = torch.where(mask.bool(), x-last_anchor[:, None, :], torch.zeros_like(x)).transpose(1, 2)
        base = last_anchor
        hm = mask.transpose(1, 2); ha = torch.log1p(age).transpose(1, 2); hv = valid.transpose(1, 2)
        calendar = mark[:, -1, 4*C:4*C+4, None].transpose(1, 2).expand(-1, C, -1)
        feat = torch.cat([rel, hm, ha, hv, base.unsqueeze(-1), torch.log1p(last_age).unsqueeze(-1),
                          last_valid.unsqueeze(-1), calendar], -1)
        ids = torch.arange(C, device=x.device); lid = self.local_id(ids)[None].expand(B, -1, -1)
        local_state = self.local_encoder(torch.cat([feat, lid], -1))
        local = base.unsqueeze(-1)+self.local_skip(rel)+self.local_head(local_state)

        def patches(a): return a.reshape(B, C, self.np, self.p)
        pm = patches(hm); counts = pm.sum(-1)
        patch_age = patches(age.transpose(1, 2))[:, :, :, -1]+torch.arange(self.np-1, -1, -1, device=x.device)[None, None, :]*self.p
        sid = self.source_id(ids)[None, :, None, :].expand(B, -1, self.np, -1)
        pid = self.patch_id(torch.arange(self.np, device=x.device))[None, None, :, :].expand(B, C, -1, -1)
        features = torch.cat([patches(rel), pm, patches(ha), patches(hv), counts.unsqueeze(-1)/self.p,
                              torch.log1p(patch_age).unsqueeze(-1), sid, pid], -1)
        source = self.patch_encoder(features); values = self.value(source)
        allowed = (counts > 0)[:, None, None, :, :] & ~torch.eye(C, dtype=torch.bool, device=x.device)[None, :, None, :, None]
        if exclude is not None:
            allowed = allowed & ~exclude[None, :, None, :, None]
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
        pooled = torch.einsum('bcgjp,bjpv->bcgv', real, values)
        correction = torch.tanh(self.correction_scale)*self.correction_head(pooled).reshape(B, C, self.h)
        forecast = (local+correction).transpose(1, 2)
        # Attention mass and candidate segments per source group, per receiver (averaged over lead groups).
        mass = torch.einsum('bcj,cjk->bck', real.sum(-1).mean(2), groups)
        cand = torch.einsum('bcj,cjk->bck', allowed[:, :, 0].to(x.dtype).sum(-1), groups)
        null = weights[..., -1].mean(2)
        return forecast, {'mass': mass, 'candidates': cand, 'null': null}
