"""A stand-in for the few datatable calls Multi-GNN's formatter makes, for Pythons that have no datatable wheel (Colab runs 3.13).

The official script's own logic still runs: only reading the file, indexing a cell, sorting by one column and writing it back are replaced.
Cells stay strings, and the sort is stable, as datatable's is. Used by mgnn.run_official_formatter only when the real package is missing.
"""
from __future__ import annotations

import csv
import sys
import types

str32 = object()  # dt.str32: read every column as text, which is all this shim does


class _Sort:
    def __init__(self, column: int):
        self.column = column


def sort(column: int) -> _Sort:
    return _Sort(column)


f = join = None  # imported by the official script and never used


class Frame:
    def __init__(self, names: list[str], rows: list[list[str]]):
        self.names, self._rows = names, rows

    @property
    def nrows(self) -> int:
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], int):
            row, col = key
            return self._rows[row][col if isinstance(col, int) else self.names.index(col)]
        if isinstance(key, tuple) and len(key) == 3 and isinstance(key[2], _Sort):
            order = sorted(range(len(self._rows)), key=lambda i: float(self._rows[i][key[2].column]))  # Python's sort is stable
            return Frame(self.names, [self._rows[i] for i in order])
        raise NotImplementedError(f"the shim supports only frame[row, column] and frame[:, :, sort(column)], not {key!r}")

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(self.names)
            writer.writerows(self._rows)


def fread(path: str, columns=None) -> Frame:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        names = [n.strip() for n in next(reader)]  # datatable trims header whitespace
        rows = list(reader)
    return Frame(names, rows)


def install() -> None:
    module = types.ModuleType("datatable")
    module.fread, module.sort, module.f, module.join, module.str32, module.Frame = fread, sort, f, join, str32, Frame
    sys.modules["datatable"] = module
