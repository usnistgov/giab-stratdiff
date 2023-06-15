import re
import gzip
import pandas as pd
import numpy as np
from multiprocessing import Pool
from os import scandir
from pathlib import Path
from pybedtools import BedTool as bt  # type: ignore
from pybedtools import cleanup  # type: ignore
from functools import partial


def list_strats(
    root: Path,
    pat: list[tuple[str, str]] = [],
    ignore: list[str] = [],
    mapper: dict[Path, Path] = {},
) -> dict[Path, Path]:
    def apply_pat(pat: list[tuple[str, str]], x: Path) -> Path:
        return (
            apply_pat(pat[1:], Path(re.sub(pat[0][0], pat[0][1], str(x))))
            if len(pat) > 0
            else x
        )

    def apply_mapper(x: Path) -> Path:
        try:
            return mapper[x]
        except KeyError:
            return apply_pat(pat, x)

    def match_any(s: Path) -> bool:
        return any(map(lambda p: re.match(p, str(s)), ignore))

    return {
        apply_mapper(p): p
        for f in scandir(root)
        if f.is_dir()
        for g in scandir(f)
        if g.name.endswith(".bed.gz")
        if not match_any(p := Path(f.name) / g.name)
    }


def read_map_file(path: Path) -> dict[Path, Path]:
    df = pd.read_table(path, header=None)[[0, 1]]
    return {Path(r[0]): Path(r[1]) for r in df.itertuples(index=False)}


def to_diagnostics(
    bed1: Path | None,
    bed2: Path | None,
    total_A: int,
    total_B: int,
    total_shared: int,
) -> dict[str, int | float | str]:
    return {
        "total_A": total_A,
        "total_B": total_B,
        "diff_AB": "" if bed1 is None or bed2 is None else total_A - total_B,
        "total_shared": total_shared,
        "shared_A": total_shared / total_A * 100 if total_A > 0 else 0,
        "shared_B": total_shared / total_B * 100 if total_B > 0 else 0,
        "bedA": str(bed1) if bed1 is not None else "",
        "bedB": str(bed2) if bed2 is not None else "",
    }


def read_bed(path: Path) -> pd.DataFrame:
    try:
        return pd.read_table(
            path,
            header=None,
            dtype={"chrom": str, "start": int, "end": int},
            names=["chrom", "start", "end"],
            comment="#",
        )
    except (gzip.BadGzipFile, pd.errors.EmptyDataError):
        return pd.DataFrame()


def bed_sum(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    else:
        return (df["end"] - df["start"]).sum()


def read_a(path: Path) -> dict[str, int | float | str]:
    return to_diagnostics(path, None, bed_sum(read_bed(path)), 0, 0)


def read_b(path: Path) -> dict[str, int | float | str]:
    return to_diagnostics(None, path, 0, bed_sum(read_bed(path)), 0)


def map_strats(
    root1: Path,
    root2: Path,
    mapper: dict[Path, Path],
    rep: list[tuple[str, str]],
    ignoreA: list[str],
    ignoreB: list[str],
) -> tuple[list[tuple[Path, Path]], list[dict[str, int | float | str]]]:
    ss1 = list_strats(root1, rep, ignoreA, mapper)
    ss2 = list_strats(root2, [], ignoreB)
    overlap = set(ss1) & set(ss2)
    mapped = sorted(
        [(ss1[f], ss2[f]) for f in overlap],
        key=lambda x: max(
            (root1 / x[0]).stat().st_size,
            (root2 / x[1]).stat().st_size,
        ),
        reverse=True,
    )
    with Pool() as p:
        a_only = p.map(read_a, [root1 / ss1[k] for k in set(ss1) - set(ss2)])
        b_only = p.map(read_b, [root2 / ss2[k] for k in set(ss2) - set(ss1)])
        return mapped, a_only + b_only


def write_df(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, header=True, sep="\t", index=False)


def compare_beds(
    root1: Path,
    root2: Path,
    outdir: Path,
    chrs: list[str],
    bed_paths: tuple[Path, Path],
) -> dict[str, int | float | str]:
    bed1 = bed_paths[0]
    bed2 = bed_paths[1]

    def write_anti(df) -> None:
        if not df.empty:
            write_df(outdir / str(bed1).replace("/", "_"), df)

    def add_bed_names(df: pd.DataFrame) -> pd.DataFrame:
        df["other_bed"] = bed2
        return df

    df1 = read_bed(root1 / bed1)
    df2 = read_bed(root2 / bed2)
    if len(chrs) > 0:
        df1 = df1[df1["chrom"].isin(chrs)]
        df2 = df2[df2["chrom"].isin(chrs)]

    if df1.empty:
        df = add_bed_names(df2.assign(**{"bed": 1, "adj": "."}))
        write_anti(df)
        return to_diagnostics(bed1, bed2, 0, bed_sum(df), 0)

    if df2.empty:
        df = add_bed_names(df1.assign(**{"bed": 0, "adj": "."}))
        write_anti(df)
        return to_diagnostics(bed1, bed2, bed_sum(df), 0, 0)

    coltypes = {
        "chrom": str,
        "start": int,
        "end": int,
        "n": int,
        "list": str,
        "inA": int,
        "inB": int,
    }

    df = (
        bt()
        .multi_intersect(i=[bt().from_dataframe(x).fn for x in [df1, df2]])
        .to_dataframe(names=[*coltypes], dtype=coltypes)
    )
    cleanup()

    # add some metadata showing if a region covered by only one bed file is
    # adjacent to a region covered by both
    df_chr = df.groupby("chrom")
    prev_adj = df["start"] == df_chr["end"].shift(1, fill_value=-1)
    next_adj = df_chr["start"].shift(-1, fill_value=-1) == df["end"]
    prev_adj_shared = prev_adj & (df_chr["n"].shift(1, fill_value=-1) == 2)
    next_adj_shared = next_adj & (df_chr["n"].shift(-1, fill_value=-1) == 2)
    df["adj"] = np.where(
        df["n"] > 1,
        "shared",
        np.where(
            prev_adj_shared & next_adj_shared,
            "<>",
            np.where(
                next_adj_shared,
                ">",
                np.where(
                    prev_adj_shared,
                    "<",
                    ".",
                ),
            ),
        ),
    )
    shared = df["adj"] == "shared"

    anti = df[~shared].copy()
    anti["bed"] = np.where(
        anti["inA"] == 1,
        0,
        np.where(anti["inB"] == 1, 1, -1),
    )
    anti = anti[["chrom", "start", "end", "bed", "adj"]].copy()
    anti["length"] = anti["end"] - anti["start"]
    anti["other_bed"] = str(bed2).replace(".bed.gz", "")

    length = df["end"] - df["start"]
    total_shared = length[shared].sum()
    total_A = length[df["inA"] == 1].sum()
    total_B = length[df["inB"] == 1].sum()

    write_anti(anti.rename(columns={"chrom": "#chrom"}))

    return to_diagnostics(bed1, bed2, total_A, total_B, total_shared)


def compare_all(
    root1: Path,
    root2: Path,
    outdir: Path,
    mapper: dict[Path, Path],
    rep: list[tuple[str, str]],
    chrs: list[str],
    ignoreA: list[str],
    ignoreB: list[str],
) -> list[str]:
    ss, nonoverlap = map_strats(root1, root2, mapper, rep, ignoreA, ignoreB)

    if len(ss) > 0:
        outdir.mkdir(parents=True, exist_ok=True)
        with Pool() as p:
            diagnostics = p.map(
                partial(compare_beds, root1, root2, outdir, chrs),
                ss,
                chunksize=1,
            )

        write_df(
            outdir / "diagnostics.tsv",
            pd.DataFrame(nonoverlap + diagnostics).sort_values(by=["bedA", "bedB"]),
        )
        return []

    else:
        return ["no overlapping beds to compare"]
