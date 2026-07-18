"""
Text normalization module for TTS processing — forked from remsky/Kokoro-FastAPI.
Handles various text formats including URLs, emails, numbers, money, and special characters.
Converts them into a format suitable for text-to-speech processing.

Upstream: https://github.com/remsky/Kokoro-FastAPI/blob/master/api/src/services/text_processing/normalizer.py
Delta: handle_markdown() function + markdown_normalization toggle in normalize_text()
"""

import math
import re
from functools import lru_cache
from typing import List, Optional, Union

import inflect
from numpy import number

# from text_to_num import text2num
from torch import mul

from ...structures.schemas import NormalizationOptions

# Constants
VALID_TLDS = [
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "mil",
    "int",
    "biz",
    "info",
    "name",
    "pro",
    "coop",
    "museum",
    "travel",
    "jobs",
    "mobi",
    "tel",
    "asia",
    "cat",
    "xxx",
    "aero",
    "arpa",
    "bg",
    "br",
    "ca",
    "cn",
    "de",
    "es",
    "eu",
    "fr",
    "in",
    "it",
    "jp",
    "mx",
    "nl",
    "ru",
    "uk",
    "us",
    "io",
    "co",
]

VALID_UNITS = {
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "in": "inch",
    "ft": "foot",
    "yd": "yard",
    "mi": "mile",  # Length
    "g": "gram",
    "kg": "kilogram",
    "mg": "milligram",  # Mass
    "s": "second",
    "ms": "millisecond",
    "min": "minutes",
    "h": "hour",  # Time
    "l": "liter",
    "ml": "mililiter",
    "cl": "centiliter",
    "dl": "deciliter",  # Volume
    "kph": "kilometer per hour",
    "mph": "mile per hour",
    "mi/h": "mile per hour",
    "m/s": "meter per second",
    "km/h": "kilometer per hour",
    "mm/s": "milimeter per second",
    "cm/s": "centimeter per second",
    "ft/s": "feet per second",
    "cm/h": "centimeter per day",  # Speed
    "°c": "degree celsius",
    "c": "degree celsius",
    "°f": "degree fahrenheit",
    "f": "degree fahrenheit",
    "k": "kelvin",  # Temperature
    "pa": "pascal",
    "kpa": "kilopascal",
    "mpa": "megapascal",
    "atm": "atmosphere",  # Pressure
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",  # Frequency
    "v": "volt",
    "kv": "kilovolt",
    "mv": "mergavolt",  # Voltage
    "a": "amp",
    "ma": "megaamp",
    "ka": "kiloamp",  # Current
    "w": "watt",
    "kw": "kilowatt",
    "mw": "megawatt",  # Power
    "j": "joule",
    "kj": "kilojoule",
    "mj": "megajoule",  # Energy
    "Ω": "ohm",
    "kΩ": "kiloohm",
    "mΩ": "megaohm",  # Resistance (Ohm)
    "f": "farad",
    "µf": "microfarad",
    "nf": "nanofarad",
    "pf": "picofarad",  # Capacitance
    "b": "bit",
    "kb": "kilobit",
    "mb": "megabit",
    "gb": "gigabit",
    "tb": "terabit",
    "pb": "petabit",  # Data size
    "kbps": "kilobit per second",
    "mbps": "megabit per second",
    "gbps": "gigabit per second",
    "tbps": "terabit per second",
    "px": "pixel",  # CSS units
}

SYMBOL_REPLACEMENTS = {
    "~": " ",
    "@": " at ",
    "#": " number ",
    "$": " dollar ",
    "%": " percent ",
    "^": " ",
    "&": " and ",
    "*": " ",
    "_": " ",
    "|": " ",
    "\\": " ",
    "/": " slash ",
    "=": " equals ",
    "+": " plus ",
}

MONEY_UNITS = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}

# Pre-compiled regex patterns for performance
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
)
URL_PATTERN = re.compile(
    r"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+(\.(?:"
    + "|".join(VALID_TLDS)
    + r"))+|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})(:[0-9]+)?([/?][^\s]*)?",
    re.IGNORECASE,
)

UNIT_PATTERN = re.compile(
    r"((?<!\w)([+-]?)(\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\s*("
    + "|".join(sorted(list(VALID_UNITS.keys()), reverse=True))
    + r"""){1}(?=[^\w\d]{1}|\b)""",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"([0-9]{1,2} ?: ?[0-9]{2}( ?: ?[0-9]{2})?)( ?(pm|am)\b)?", re.IGNORECASE
)

MONEY_PATTERN = re.compile(
    r"(-?)(["
    + "".join(MONEY_UNITS.keys())
    + r"])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Markdown normalization patterns (added by navi-os fork)
# ---------------------------------------------------------------------------
# Order matters: fenced code blocks first, then inline, then structural.
# Bold before italic to avoid partial matches on **.
# A fence opens at line start (optionally inside a blockquote); an unclosed
# fence (streaming/truncated chunks) extends to end of input — strip the
# fence line, keep the content.
_MD_FENCED_CODE = re.compile(
    r"^[ \t]*(>[ \t]*)?```[^\n]*\n?(.*?)(?:```|\Z)", re.DOTALL | re.MULTILINE
)
# Blockquote marker to dedent from fenced content when the fence itself was
# blockquoted ('> ```' ... '> code' ... '> ```').
_MD_BLOCKQUOTE_DEDENT = re.compile(r"^>[ \t]?", re.MULTILINE)
# Placeholder restore patterns (Phase 3) — input NULs are stripped up front,
# so these can only match placeholders this module minted.
_MD_CB_PLACEHOLDER = re.compile("\x00CB(\\d+)\x00")
_MD_IC_PLACEHOLDER = re.compile("\x00IC(\\d+)\x00")
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
# URL part tolerates one level of balanced parens, e.g. wiki/Foo_(bar) —
# enough for real-world URLs without over-matching past the closing ).
# Link text and URL are length-bounded like emphasis: on '[' floods the
# unbounded '[^\]]*' scanned to end-of-string from every position (O(n^2)).
_MD_URL_PART = r"(?:[^()]|\([^()]*\)){1,2000}"
_MD_IMAGE = re.compile(r"!\[([^\]]{0,800})\]\(" + _MD_URL_PART + r"\)")
# Link text may be empty ('[](url)') — the whole link then drops entirely.
_MD_LINK = re.compile(r"\[([^\]]{0,800})\]\(" + _MD_URL_PART + r"\)")
# Emphasis content: spans single newlines (soft wraps) but never a blank
# line (paragraph break) — CommonMark-ish. Bounded at 600 chars so a flood
# of unclosed markers scans O(n·600) instead of O(n^2); longer "spans" are
# not real emphasis and simply stay unstripped.
_MD_EMPHASIS_CONTENT = r"(?:[^\n]|\n(?![ \t]*\n)){1,600}?"
_MD_BOLD = re.compile(r"\*\*(" + _MD_EMPHASIS_CONTENT + r")\*\*")
# Bold with __ — word-boundary-aware like the underscore italic below.
_MD_BOLD_UNDER = re.compile(r"(?<!\w)__(" + _MD_EMPHASIS_CONTENT + r")__(?!\w)")
# Italic with * — only match when not preceded/followed by word chars to avoid
# false positives inside URLs or filenames.
_MD_ITALIC_STAR = re.compile(r"(?<!\w)\*(" + _MD_EMPHASIS_CONTENT + r")\*(?!\w)")
# Italic with _ — word-boundary-aware so snake_case stays intact.
_MD_ITALIC_UNDER = re.compile(r"(?<!\w)_(" + _MD_EMPHASIS_CONTENT + r")_(?!\w)")
_MD_STRIKETHROUGH = re.compile(r"~~(" + _MD_EMPHASIS_CONTENT + r")~~")
# ATX headings, tolerating indentation and an optional closing hash
# sequence ('## H ##' -> 'H') — but '#' glued to a word stays ('C#').
_MD_HEADING = re.compile(
    r"^[ \t]*#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$", re.MULTILINE
)
_MD_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_MD_HORIZONTAL_RULE = re.compile(r"^(?:---+|\*\*\*+|___+)\s*$", re.MULTILINE)
_MD_UNORDERED_LIST = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)
_MD_ORDERED_LIST = re.compile(r"^(\s*)\d+\.\s+", re.MULTILINE)
# Table separator rows like |---|---|. Detected with a single ambiguity-free
# character class plus a substring check: the old '-{3,}[\s:|-]*' form had two
# adjacent quantifiers both matching '-', giving O(n^2) backtracking on long
# dash lines.
_MD_TABLE_SEP_CHARS = re.compile(r"[ \t:|-]*")


def _is_table_sep_row(line: str) -> bool:
    return "---" in line and _MD_TABLE_SEP_CHARS.fullmatch(line) is not None


def handle_markdown(text: str) -> str:
    """Strip markdown formatting so TTS reads content, not syntax.

    Designed as a pre-processing pass: runs BEFORE email/URL normalization
    so that markdown link syntax ``[text](url)`` is reduced to ``text``
    before the URL handler sees bare URLs.
    """
    # --- Phase 1: Protect code content from markdown stripping ---
    # Drop any NUL bytes first: they can't be voiced anyway, and it makes the
    # \x00-delimited placeholders below collision-proof against input that
    # happens to contain literal placeholder-looking text.
    text = text.replace("\x00", "")
    # Normalize line endings: every MULTILINE anchor and blank-line lookahead
    # below assumes \n; raw \r defeats them (CRLF headings leak '##', CRLF
    # blank lines stop acting as paragraph breaks for emphasis).
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Extract fenced code blocks and inline code into placeholders so that
    # markdown syntax inside code (e.g. **bold**) is preserved literally.
    code_blocks: list[str] = []

    def _save_fenced(m: re.Match) -> str:
        content = m.group(2)
        if m.group(1):
            # Fence was blockquoted — dedent the '> ' markers from content.
            content = _MD_BLOCKQUOTE_DEDENT.sub("", content)
        code_blocks.append(content)
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = _MD_FENCED_CODE.sub(_save_fenced, text)

    inline_codes: list[str] = []

    def _save_inline(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = _MD_INLINE_CODE.sub(_save_inline, text)

    # --- Phase 2: Strip markdown formatting ---
    # Images — keep alt text
    text = _MD_IMAGE.sub(r"\1", text)
    # Links — keep link text
    text = _MD_LINK.sub(r"\1", text)
    # Bold (** and __ before single-char markers to avoid partial match)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_BOLD_UNDER.sub(r"\1", text)
    # Italic (* and _)
    text = _MD_ITALIC_STAR.sub(r"\1", text)
    text = _MD_ITALIC_UNDER.sub(r"\1", text)
    # Strikethrough
    text = _MD_STRIKETHROUGH.sub(r"\1", text)
    # List markers — strip marker, keep text. Must run before headings so
    # '- # Heading' loses both the marker and the hashes.
    text = _MD_UNORDERED_LIST.sub(r"\1", text)
    text = _MD_ORDERED_LIST.sub(r"\1", text)
    # Headings — strip leading hashes (and any closing hash sequence)
    text = _MD_HEADING.sub(r"\1", text)
    # Blockquotes — strip leading >
    text = _MD_BLOCKQUOTE.sub("", text)
    # Horizontal rules — remove entire line
    text = _MD_HORIZONTAL_RULE.sub("", text)
    # Tables — remove separator rows, replace pipes with spaces on table
    # rows. A row must have 2+ pipes AND either be pipe-anchored (leading or
    # trailing |) or sit next to a separator row; prose pipes like
    # "either a | b | c works" stay intact.
    # Escaped pipes (\|) are literal content — hide them from the table
    # logic, then restore as bare pipes at the end.
    text = text.replace("\\|", "\x00EP\x00")
    lines = text.split("\n")
    is_sep_row = [_is_table_sep_row(line) for line in lines]
    for i, line in enumerate(lines):
        if is_sep_row[i]:
            lines[i] = ""
            continue
        if line.count("|") < 2:
            continue
        stripped = line.strip()
        pipe_anchored = stripped.startswith("|") or stripped.endswith("|")
        near_sep_row = (i > 0 and is_sep_row[i - 1]) or (
            i + 1 < len(lines) and is_sep_row[i + 1]
        )
        if pipe_anchored or near_sep_row:
            lines[i] = line.replace("|", " ")
    text = "\n".join(lines).replace("\x00EP\x00", "|")

    # --- Phase 3: Restore protected code content ---
    # Single-pass sub keyed on the placeholder index: a per-item str.replace
    # loop is O(placeholders x len(text)) — quadratic on fence/backtick floods.
    if code_blocks:
        text = _MD_CB_PLACEHOLDER.sub(lambda m: code_blocks[int(m.group(1))], text)
    if inline_codes:
        text = _MD_IC_PLACEHOLDER.sub(lambda m: inline_codes[int(m.group(1))], text)

    return text


# ---------------------------------------------------------------------------
# Upstream handlers (unchanged from v0.2.4)
# ---------------------------------------------------------------------------

INFLECT_ENGINE = inflect.engine()


def handle_units(u: re.Match[str]) -> str:
    """Converts units to their full form"""
    unit_string = u.group(6).strip()
    unit = unit_string

    if unit_string.lower() in VALID_UNITS:
        unit = VALID_UNITS[unit_string.lower()].split(" ")

        # Handles the B vs b case
        if unit[0].endswith("bit"):
            b_case = unit_string[min(1, len(unit_string) - 1)]
            if b_case == "B":
                unit[0] = unit[0][:-3] + "byte"

        number = u.group(1).strip()
        unit[0] = INFLECT_ENGINE.no(unit[0], number)
    return " ".join(unit)


def conditional_int(number: float, threshold: float = 0.00001):
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def translate_multiplier(multiplier: str) -> str:
    """Translate multiplier abrevations to words"""

    multiplier_translation = {
        "k": "thousand",
        "m": "million",
        "b": "billion",
        "t": "trillion",
    }
    if multiplier.lower() in multiplier_translation:
        return multiplier_translation[multiplier.lower()]
    return multiplier.strip()


def split_four_digit(number: float):
    part1 = str(conditional_int(number))[:2]
    part2 = str(conditional_int(number))[2:]
    return f"{INFLECT_ENGINE.number_to_words(part1)} {INFLECT_ENGINE.number_to_words(part2)}"


def handle_numbers(n: re.Match[str]) -> str:
    number = n.group(2)

    try:
        number = float(number)
    except:
        return n.group()

    if n.group(1) == "-":
        number *= -1

    multiplier = translate_multiplier(n.group(3))

    number = conditional_int(number)
    if multiplier != "":
        multiplier = f" {multiplier}"
    else:
        if (
            number % 1 == 0
            and len(str(number)) == 4
            and number > 1500
            and number % 1000 > 9
        ):
            return split_four_digit(number)

    return f"{INFLECT_ENGINE.number_to_words(number)}{multiplier}"


def handle_money(m: re.Match[str]) -> str:
    """Convert money expressions to spoken form"""

    bill, coin = MONEY_UNITS[m.group(2)]

    number = m.group(3)

    try:
        number = float(number)
    except:
        return m.group()

    if m.group(1) == "-":
        number *= -1

    multiplier = translate_multiplier(m.group(4))

    if multiplier != "":
        multiplier = f" {multiplier}"

    if number % 1 == 0 or multiplier != "":
        text_number = f"{INFLECT_ENGINE.number_to_words(conditional_int(number))}{multiplier} {INFLECT_ENGINE.plural(bill, count=number)}"
    else:
        sub_number = int(str(number).split(".")[-1].ljust(2, "0"))

        text_number = f"{INFLECT_ENGINE.number_to_words(int(math.floor(number)))} {INFLECT_ENGINE.plural(bill, count=number)} and {INFLECT_ENGINE.number_to_words(sub_number)} {INFLECT_ENGINE.plural(coin, count=sub_number)}"

    return text_number


def handle_decimal(num: re.Match[str]) -> str:
    """Convert decimal numbers to spoken form"""
    a, b = num.group().split(".")
    return " point ".join([a, " ".join(b)])


def handle_email(m: re.Match[str]) -> str:
    """Convert email addresses into speakable format"""
    email = m.group(0)
    parts = email.split("@")
    if len(parts) == 2:
        user, domain = parts
        domain = domain.replace(".", " dot ")
        return f"{user} at {domain}"
    return email


def handle_url(u: re.Match[str]) -> str:
    """Make URLs speakable by converting special characters to spoken words"""
    if not u:
        return ""

    url = u.group(0).strip()

    # Handle protocol first
    url = re.sub(
        r"^https?://",
        lambda a: "https " if "https" in a.group() else "http ",
        url,
        flags=re.IGNORECASE,
    )
    url = re.sub(r"^www\.", "www ", url, flags=re.IGNORECASE)

    # Handle port numbers before other replacements
    url = re.sub(r":(\d+)(?=/|$)", lambda m: f" colon {m.group(1)}", url)

    # Split into domain and path
    parts = url.split("/", 1)
    domain = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    # Handle dots in domain
    domain = domain.replace(".", " dot ")

    # Reconstruct URL
    if path:
        url = f"{domain} slash {path}"
    else:
        url = domain

    # Replace remaining symbols with words
    url = url.replace("-", " dash ")
    url = url.replace("_", " underscore ")
    url = url.replace("?", " question-mark ")
    url = url.replace("=", " equals ")
    url = url.replace("&", " ampersand ")
    url = url.replace("%", " percent ")
    url = url.replace(":", " colon ")  # Handle any remaining colons
    url = url.replace("/", " slash ")  # Handle any remaining slashes

    # Clean up extra spaces
    return re.sub(r"\s+", " ", url).strip()


def handle_phone_number(p: re.Match[str]) -> str:
    p = list(p.groups())

    country_code = ""
    if p[0] is not None:
        p[0] = p[0].replace("+", "")
        country_code += INFLECT_ENGINE.number_to_words(p[0])

    area_code = INFLECT_ENGINE.number_to_words(
        p[2].replace("(", "").replace(")", ""), group=1, comma=""
    )

    telephone_prefix = INFLECT_ENGINE.number_to_words(p[3], group=1, comma="")

    line_number = INFLECT_ENGINE.number_to_words(p[4], group=1, comma="")

    return ",".join([country_code, area_code, telephone_prefix, line_number])


def handle_time(t: re.Match[str]) -> str:
    t = t.groups()

    time_parts = t[0].split(":")

    numbers = []
    numbers.append(INFLECT_ENGINE.number_to_words(time_parts[0].strip()))

    minute_number = INFLECT_ENGINE.number_to_words(time_parts[1].strip())
    if int(time_parts[1]) < 10:
        if int(time_parts[1]) != 0:
            numbers.append(f"oh {minute_number}")
    else:
        numbers.append(minute_number)

    half = ""
    if len(time_parts) > 2:
        seconds_number = INFLECT_ENGINE.number_to_words(time_parts[2].strip())
        second_word = INFLECT_ENGINE.plural("second", int(time_parts[2].strip()))
        numbers.append(f"and {seconds_number} {second_word}")
    else:
        if t[2] is not None:
            half = " " + t[2].strip()
        else:
            if int(time_parts[1]) == 0:
                numbers.append("o'clock")

    return " ".join(numbers) + half


def normalize_text(text: str, normalization_options: NormalizationOptions) -> str:
    """Normalize text for TTS processing"""

    # Handle markdown formatting FIRST — strips syntax like **bold**, [link](url),
    # ```code```, etc. before other handlers see the raw symbols.
    # Must run before email/URL normalization: markdown links contain URLs that
    # should be reduced to link text, not spelled out.
    if normalization_options.markdown_normalization:
        text = handle_markdown(text)

    # Handle email addresses first if enabled
    if normalization_options.email_normalization:
        text = EMAIL_PATTERN.sub(handle_email, text)

    # Handle URLs if enabled
    if normalization_options.url_normalization:
        text = URL_PATTERN.sub(handle_url, text)

    # Pre-process numbers with units if enabled
    if normalization_options.unit_normalization:
        text = UNIT_PATTERN.sub(handle_units, text)

    # Replace optional pluralization
    if normalization_options.optional_pluralization_normalization:
        text = re.sub(r"\(s\)", "s", text)

    # Replace phone numbers:
    if normalization_options.phone_normalization:
        text = re.sub(
            r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})",
            handle_phone_number,
            text,
        )

    # Replace quotes and brackets (additional cleanup)
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace("«", chr(8220)).replace("»", chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')

    # Handle CJK punctuation and some non standard chars
    for a, b in zip("、。！，：；？–", ",.!,:;?-"):
        text = text.replace(a, b + " ")

    # Handle simple time in the format of HH:MM:SS (am/pm)
    text = TIME_PATTERN.sub(
        handle_time,
        text,
    )

    # Clean up whitespace
    text = re.sub(r"[^\S \n]", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"(?<=\n) +(?=\n)", "", text)

    # Handle special characters that might cause audio artifacts first
    # Replace newlines with spaces (or pauses if needed)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Handle titles and abbreviations
    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)

    # Handle common words
    text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)

    # Handle numbers and money BEFORE replacing special characters
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    text = MONEY_PATTERN.sub(
        handle_money,
        text,
    )

    text = NUMBER_PATTERN.sub(handle_numbers, text)

    text = re.sub(r"\d*\.\d+", handle_decimal, text)

    # Handle other problematic symbols AFTER money/number processing
    if normalization_options.replace_remaining_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)

    # Handle various formatting
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    text = re.sub(r"(?<=\d)S", " S", text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", "s", text)
    text = re.sub(
        r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text
    )
    text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)

    text = re.sub(r"\s{2,}", " ", text)

    return text
