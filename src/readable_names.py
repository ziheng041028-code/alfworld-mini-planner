from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict

COORD_PATTERN = re.compile(
    r"^(?P<obj>[A-Za-z0-9]+)"
    r"_bar__(?P<xsign>minus|plus)_(?P<xint>\d+)_dot_(?P<xfrac>\d+)"
    r"_bar__(?P<ysign>minus|plus)_(?P<yint>\d+)_dot_(?P<yfrac>\d+)"
    r"_bar__(?P<zsign>minus|plus)_(?P<zint>\d+)_dot_(?P<zfrac>\d+)$"
)

INLINE_OBJECT_PATTERN = re.compile(
    r"[A-Za-z0-9]+_bar__(?:minus|plus)_\d+_dot_\d+"
    r"_bar__(?:minus|plus)_\d+_dot_\d+"
    r"_bar__(?:minus|plus)_\d+_dot_\d+"
)


def _to_float(sign: str, integer: str, frac: str) -> str:
    value = f"{int(integer)}.{frac}"
    prefix = "-" if sign == "minus" else "+"
    return f"{prefix}{value}"


def decode_object_id(name: str) -> str:
    """
    例如：
    drawer_bar__minus_00_dot_51_bar__plus_00_dot_72_bar__minus_00_dot_85
    ->
    drawer@(-0.51,+0.72,-0.85)
    """
    m = COORD_PATTERN.match(name)
    if not m:
        return name

    obj = m.group("obj")
    x = _to_float(m.group("xsign"), m.group("xint"), m.group("xfrac"))
    y = _to_float(m.group("ysign"), m.group("yint"), m.group("yfrac"))
    z = _to_float(m.group("zsign"), m.group("zint"), m.group("zfrac"))
    return f"{obj}@({x},{y},{z})"


class AliasMapper:
    def __init__(self) -> None:
        self.full_to_alias: Dict[str, str] = {}
        self.counter = defaultdict(int)

    def alias(self, name: str) -> str:
        if name in self.full_to_alias:
            return self.full_to_alias[name]

        decoded = decode_object_id(name)
        base = decoded.split("@")[0]
        self.counter[base] += 1
        alias = f"{base}#{self.counter[base]}"
        self.full_to_alias[name] = alias
        return alias


def prettify_text(text: str) -> str:
    """
    用 regex 在整段文本里替换，保留原有标点、换行和句子结构。
    """

    def repl(match: re.Match) -> str:
        raw = match.group(0)
        return decode_object_id(raw)

    return INLINE_OBJECT_PATTERN.sub(repl, text)


def prettify_with_alias(text: str, mapper: AliasMapper) -> str:
    def repl(match: re.Match) -> str:
        raw = match.group(0)
        return mapper.alias(raw)

    return INLINE_OBJECT_PATTERN.sub(repl, text)
