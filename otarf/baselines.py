"""Helpers for the UNCHANGED official SOFTS / TimeFilter code (cloned under external/)."""

import argparse
import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch

from .io import read_json, sha256


def verify_official_source(repository, name, lock_file):
    lock = read_json(lock_file)[name]
    for relative, digest in lock.get("python_sha256", {}).items():
        if sha256(Path(repository) / relative) != digest:
            raise RuntimeError(f"Modified official code: {name}/{relative}")
    return lock["commit"]


def official_parser(repository):
    # Only the argparse declarations of run.py are evaluated; importing run.py would start training.
    tree = ast.parse((Path(repository) / "run.py").read_text())
    parser = argparse.ArgumentParser()
    declarations = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "parser"
        and n.func.attr == "add_argument"
    ]
    for node in sorted(declarations, key=lambda n: n.lineno):
        eval(
            compile(ast.Expression(node), str(Path(repository) / "run.py"), "eval"),
            {
                "__builtins__": {},
                "parser": parser,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
            },
        )
    return parser


def make_arguments(
    name, run_id, horizon, config, meta, directory, output, repository, task_name=True
):
    values = {
        "task_name": "long_term_forecast",
        "is_training": 1,
        "model_id": run_id,
        "model": name,
        "data": "custom",
        "root_path": str(directory),
        "data_path": "data.csv",
        "features": "M",
        "target": meta["official_target"],
        "freq": "h",
        "seq_len": config["seq_len"],
        "label_len": config["label_len"],
        "pred_len": horizon,
        "enc_in": meta["channels"],
        "dec_in": meta["channels"],
        "c_out": meta["channels"],
        "num_workers": config["num_workers"],
        "checkpoints": str(Path(output) / "checkpoints"),
        "itr": 1,
        **config["common"],
        **config["model_args"][name],
    }
    if not task_name:  # iTransformer's run.py has no --task_name argument
        values.pop("task_name")
        values["des"] = "Exp"
    argv = []
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        else:
            argv += ["--" + key, str(value)]
    return official_parser(repository).parse_args(argv)


class OfficialRecipeModel(torch.nn.Module):
    """Exposes model(x, mark) for an unchanged official class.

    With indicators=True the observation mask and log(1+age) of every channel are appended as extra
    variates (the official models treat variates as tokens, so no model code changes are needed).
    """

    def __init__(self, name, official, masks, channels, label_len, horizon, indicators):
        super().__init__()
        if name not in ("SOFTS", "TimeFilter"):
            raise ValueError("Unsupported official baseline")
        self.name, self.official, self.masks = name, official, masks
        self.c, self.label_len, self.h, self.indicators = (
            channels,
            label_len,
            horizon,
            indicators,
        )
        self.aux_loss = None

    def model_input(self, x, mark):
        if not self.indicators:
            return x
        c = self.c
        # mark layout: [mask c | age c | anchor-valid c | anchor c | calendar 4]
        return torch.cat(
            [x, mark[..., :c], torch.log1p(mark[..., c : 2 * c].clamp_min(0))], dim=-1
        )

    def forward(self, x, mark, *ignored):
        inp = self.model_input(x, mark)
        if self.name == "TimeFilter":
            out, aux = self.official(inp, self.masks, is_training=self.training)
            self.aux_loss = aux
        else:
            calendar = mark[..., 4 * self.c :]
            decoder = torch.cat(
                [
                    inp[:, -self.label_len :],
                    inp.new_zeros(inp.shape[0], self.h, inp.shape[2]),
                ],
                dim=1,
            )
            out = self.official(inp, calendar, decoder, None)
            self.aux_loss = inp.new_zeros(())
        return out[:, -self.h :, : self.c]


# ----------------------------------------------------------------------------------------------
# iTransformer and CRIB (added for the extended comparison; official code is again used unchanged)
# ----------------------------------------------------------------------------------------------
def load_crib_class(repo):
    """Import CRIB without running TSL_models/__init__.py, which imports every bundled baseline and
    therefore extra packages. The four files CRIB needs are loaded unchanged."""
    package = types.ModuleType("TSL_models")
    package.__path__ = [str(Path(repo) / "TSL_models")]
    sys.modules["TSL_models"] = package
    return importlib.import_module("TSL_models.CRIB").CRIB


def crib_arguments(repo, channels, horizon, seq_len, device, config):
    """Official CRIB argument parser (utils/args.py is loaded by path; utils/__init__.py is not run)."""
    spec = importlib.util.spec_from_file_location(
        "crib_official_args", Path(repo) / "utils/args.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mc = config["model_args"]["CRIB"]
    argv = ["--model", "CRIB", "--dataset", "BeijingAir", "--seq_len", str(seq_len), "--pred_len", str(horizon)]
    for key in ("batch_size", "patch_len", "learning_rate", "model_dim", "loss_type"):
        argv += ["--" + key, str(mc[key])]
    argv += ["--train_epochs", str(config["common"]["train_epochs"])]
    args = module.parse_args(argv)
    # modify_args fixes var_num by dataset name; the prepared datasets have their own channel counts.
    args.var_num = args.enc_in = args.dec_in = args.c_out = channels
    args.device = device
    return args


def itransformer_class(repo):
    sys.path.insert(0, str(repo))
    from model.iTransformer import Model

    return Model


class ExtendedRecipeModel(torch.nn.Module):
    """model(x, mark) around the unchanged official iTransformer or CRIB class.

    iTransformer: the filled series plus the observation mask and log(1+age) of every channel as extra
    variates (3C variates), calendar features as x_mark_enc.
    CRIB: the authors' input convention. The standardized series is multiplied by the observation mask
    (unobserved entries become 0) and split into patches [B, P, C, L]; no extra variates, because CRIB
    attends jointly over all patch x variate tokens (a 3C input would make attention about nine times
    larger), so CRIB does not receive the age of the last observation. Its two auxiliary official terms
    (consistency between the two views, KL) are exposed as aux_loss with the official weights.
    """

    def __init__(self, name, official, channels, horizon, patch_len=8, consistency_weight=1.0, kl_weight=1e-6):
        super().__init__()
        if name not in ("iTransformer", "CRIB"):
            raise ValueError("Unsupported official baseline " + str(name))
        self.name, self.official = name, official
        self.c, self.h, self.patch_len = channels, horizon, patch_len
        self.consistency_weight, self.kl_weight = consistency_weight, kl_weight
        self.aux_loss = None

    def forward(self, x, mark, *ignored):
        c = self.c
        # mark layout: [mask c | age c | anchor-valid c | anchor c | calendar 4]
        if self.name == "iTransformer":
            inp = torch.cat(
                [x, mark[..., :c], torch.log1p(mark[..., c : 2 * c].clamp_min(0))], dim=-1
            )
            out = self.official(inp, mark[..., 4 * c :], inp, None)
            self.aux_loss = inp.new_zeros(())
            return out[:, -self.h :, :c]
        b, t, _ = x.shape
        if t % self.patch_len:
            raise ValueError("Sequence length must be divisible by the CRIB patch length")
        zero_filled = x * mark[..., :c]
        patches = (
            zero_filled.permute(0, 2, 1)
            .reshape(b, c, t // self.patch_len, self.patch_len)
            .permute(0, 2, 1, 3)
        )
        enc1, _, enc2, _, preds, _, kl = self.official(patches, None, test_flag=not self.training)
        if self.training:
            self.aux_loss = (
                self.consistency_weight * torch.nn.functional.mse_loss(enc1, enc2)
                + self.kl_weight * kl
            )
        else:
            self.aux_loss = x.new_zeros(())
        return preds


def build_recipe_baseline(name, repo, config, meta, directory, out, horizon, device, seq_len=96, label_len=48,
                          window_norm=True):
    """Wrapped official model with the OTARF data interface, for any of the four retrained baselines
    (mask and age variates for SOFTS / TimeFilter / iTransformer, the authors' zero-filled input for CRIB).
    window_norm=False builds the same model with its window normalization switched off (otarf.nonorm)."""
    from .nonorm import disable_window_norm

    c, h = meta["channels"], horizon
    repo = Path(repo)
    if name in ("SOFTS", "TimeFilter"):
        args = make_arguments(name, f"{meta['name']}_{name}_h{h}", h, config, meta, directory, out, repo)
        args.enc_in = args.dec_in = args.c_out = 3 * c
        args.use_gpu, args.use_multi_gpu, args.gpu = True, False, 0
        if not window_norm and name == "SOFTS":
            args.use_norm = 0
        sys.path.insert(0, str(repo))
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

        exp = Exp_Long_Term_Forecast(args)
        if not window_norm:
            disable_window_norm(name, exp.model)
        return OfficialRecipeModel(name, exp.model, getattr(exp, "masks", None), c, label_len, h, True).to(device)
    if name == "iTransformer":
        args = make_arguments(name, f"{meta['name']}_{name}_h{h}", h, config, meta, directory, out, repo, task_name=False)
        args.enc_in = args.dec_in = args.c_out = 3 * c
        args.use_gpu, args.use_multi_gpu, args.gpu = True, False, 0
        for key, value in config["model_args"][name].items():
            if getattr(args, key) != value:
                raise ValueError("Official argument differs from the configuration: " + key)
        if not window_norm:
            args.use_norm = 0
        official = itransformer_class(repo)(args)
        if not window_norm:
            disable_window_norm(name, official)
        return ExtendedRecipeModel(name, official, c, h).to(device)
    if name == "CRIB":
        args = crib_arguments(repo, c, h, seq_len, device, config)
        for key, value in config["model_args"][name].items():
            if getattr(args, key) != value:
                raise ValueError("Official argument differs from the configuration: " + key)
        official = load_crib_class(repo)(args)
        if not window_norm:
            disable_window_norm(name, official)
        return ExtendedRecipeModel(
            name, official, c, h, patch_len=args.patch_len,
            consistency_weight=args.Consis_weight, kl_weight=args.KL_weight,
        ).to(device)
    raise ValueError("Unsupported baseline " + str(name))
