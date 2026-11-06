"""Regression tests for the first Colab pilot of NB2, which hit two faults in how the official code is called.

The official code is never imported here, so these tests use a stand-in that copies only the relevant argument definitions.
"""
import argparse

import polars as pl

import amlh.fraudgt as fg
import amlh.mgnn as mg
from amlh.fraudgt_driver import format_beside_raw
from test_notebooks import restore_globals  # noqa: F401 (autouse fixture)


def _official_like_parser():
    """The argument definitions of Multi-GNN's util.create_parser that matter here. num_neighs has no type, exactly as in the official file."""
    p = argparse.ArgumentParser()
    for flag in ("--reverse_mp", "--ports", "--ego", "--emlps", "--testing", "--tqdm"):
        p.add_argument(flag, action="store_true")
    p.add_argument("--batch_size", default=8192, type=int)
    p.add_argument("--n_epochs", default=100, type=int)
    p.add_argument("--num_neighs", nargs="+", default=[100, 100])
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--data", required=True, type=str)
    p.add_argument("--model", required=True, type=str)
    return p


def test_neighbour_counts_reach_the_sampler_as_integers(monkeypatch):
    monkeypatch.setattr(mg, "_official", lambda: (None, None, None, _official_like_parser))
    args = mg._args({"folder": "Small_HI", "model": "gin", "seed": 1})
    assert args.num_neighs == [100, 100] and all(isinstance(x, int) for x in args.num_neighs)
    args = mg._args({"folder": "Small_HI", "model": "pna", "seed": 2, "num_neighs": [50, 20], "n_epochs": 2})
    assert args.num_neighs == [50, 20] and all(isinstance(x, int) for x in args.num_neighs) and args.n_epochs == 2 and args.seed == 2
    # What the stand-in would have returned without the conversion, which is what the real sampler rejected.
    raw = _official_like_parser().parse_args(["--data", "d", "--model", "m", "--num_neighs", "100", "100"])
    assert raw.num_neighs == ["100", "100"]


def test_fraudgt_finds_its_raw_file_where_its_loader_looks(tmp_path):
    raw = tmp_path / "in" / "HI-Small_Trans.csv"
    raw.parent.mkdir()
    raw.write_text("Timestamp\n")
    data_dir = tmp_path / "work" / "HI-Small"
    fg.link_raw(raw, data_dir)
    # The loader joins dataset.dir with "AML" and the dataset class reads <root>/<csv name>, as in the failing traceback.
    assert (data_dir / "AML" / "HI-Small_Trans.csv").read_text() == "Timestamp\n"
    fg.link_raw(raw, data_dir)  # idempotent


def test_formatted_file_lands_beside_the_raw_file(small_data, tmp_path):
    root = tmp_path / "AML"
    root.mkdir()
    raw = small_data / "raw" / "HI-Small_Trans.csv"
    link = root / raw.name
    link.write_bytes(raw.read_bytes())
    format_beside_raw(str(link))
    out = root / "formatted_transactions.csv"
    assert out.exists() and pl.read_csv(out).columns[:3] == ["EdgeID", "from_id", "to_id"]
    assert not (tmp_path / "formatted_transactions.csv").exists()


def test_ego_transform_works_when_the_base_class_demands_forward():
    """torch-geometric 2.7.0 made BaseTransform abstract on `forward`; the official AddEgoIds defines only __call__, so it cannot be built there.
    The adapter must build it, and must delegate to the official __call__ so behaviour is unchanged."""
    import abc
    import types

    class AbstractBase(abc.ABC):
        @abc.abstractmethod
        def forward(self, data):
            ...

    class AddEgoIds(AbstractBase):   # like the official class: __call__ only
        def __call__(self, data):
            return ("official", data)

    stand_in = types.SimpleNamespace(AddEgoIds=AddEgoIds)
    try:
        AddEgoIds()
        raise AssertionError("the stand-in should be abstract")
    except TypeError:
        pass
    t = mg.ego_transform(stand_in)
    assert t("batch") == ("official", "batch")

    class Plain:   # older torch-geometric: nothing abstract, the class is used as it is
        def __call__(self, data):
            return data
    assert type(mg.ego_transform(types.SimpleNamespace(AddEgoIds=Plain))) is Plain


def test_fraudgt_dataloader_workers_default_to_zero_only_on_windows(monkeypatch, tmp_path):
    """PC-17 found FraudGT's DataLoader workers crash silently on Windows (D-022). The default is 0 there and the official 2 elsewhere,
    and an explicit cfg value always wins."""
    def workers(platform, cfg=None):
        monkeypatch.setattr(fg.sys, "platform", platform)
        out = fg.overrides({"seed": 1, **(cfg or {})}, tmp_path, tmp_path)
        return out[out.index("num_workers") + 1]
    assert workers("win32") == "0"
    assert workers("linux") == "2"
    assert workers("win32", {"num_workers": 3}) == "3"
