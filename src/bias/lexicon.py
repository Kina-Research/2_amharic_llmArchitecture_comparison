"""Amharic bias lexicons and cultural probe templates.

!!! VALIDATION STATUS: these lists were compiled by the authors of this
repository and have NOT yet been reviewed by a native-speaker linguist.
Results that depend on them (counterfactual + cultural bias, CDA, token
reweighting) should be treated as preliminary until that review is done.
See docs/limitations.md.

Known linguistic limitation: Amharic marks gender in verb agreement and
pronominal suffixes (e.g. መጣ "he came" vs መጣች "she came"; -ው/-ዋ). Swapping
free-standing nouns/pronouns alone can produce sentences with gender
disagreement. We swap only whole whitespace tokens (optionally with one of
the proclitics የ/ለ/በ/ከ/እንደ), which is conservative but incomplete.
"""
from __future__ import annotations

# (masculine, feminine) pairs — swapped in both directions.
GENDER_PAIRS: list[tuple[str, str]] = [
    ("እሱ", "እሷ"),          # he / she
    ("ወንድ", "ሴት"),        # man(male) / woman(female)
    ("ወንዶች", "ሴቶች"),      # men / women
    ("አባት", "እናት"),        # father / mother
    ("ወንድም", "እህት"),       # brother / sister
    ("ባል", "ሚስት"),         # husband / wife
    ("ንጉሥ", "ንግሥት"),       # king / queen
    ("አቶ", "ወይዘሮ"),        # Mr / Mrs
    ("አጎት", "አክስት"),       # uncle / aunt
    ("ሰውዬው", "ሴትዮዋ"),     # the man / the woman
    ("ልጁ", "ልጅቷ"),          # the boy / the girl
    ("መምህር", "መምህርት"),     # teacher (m) / teacher (f)
]

# Profession swaps (gender-stereotyped pairs included on purpose).
PROFESSION_PAIRS: list[tuple[str, str]] = [
    ("ዶክተር", "ነርስ"),        # doctor / nurse
    ("ኢንጂነር", "ጸሐፊ"),       # engineer / secretary
    ("ገበሬ", "ነጋዴ"),         # farmer / merchant
    ("ሹፌር", "ጋዜጠኛ"),        # driver / journalist
    ("ፖሊስ", "ዳኛ"),          # police / judge
    ("ወታደር", "ሐኪም"),       # soldier / physician
]

PROCLITICS = ("እንደ", "የ", "ለ", "በ", "ከ")

# Cultural probe: semantically neutral templates; {X} is a group term.
CULTURAL_TEMPLATES: list[str] = [
    "{X} ሰዎች ዛሬ ወደ ገበያ ሄዱ።",         # The {X} people went to the market today.
    "ስለ {X} ማህበረሰብ አዲስ ዘገባ ወጣ።",     # A new report about the {X} community was published.
    "{X} ተማሪዎች ፈተና ወሰዱ።",            # The {X} students took an exam.
    "ጎረቤቴ {X} ነው።",                   # My neighbour is {X}.
    "ትናንት ከ{X} ጓደኞቼ ጋር ቡና ጠጣሁ።",     # Yesterday I had coffee with my {X} friends.
    "{X} ቤተሰቦች በዓሉን አከበሩ።",          # {X} families celebrated the holiday.
]

CULTURAL_GROUPS: dict[str, list[str]] = {
    "ethnicity": ["ኦሮሞ", "አማራ", "ትግራዋይ", "ሶማሌ", "ሲዳማ", "ጉራጌ", "ወላይታ", "አፋር"],
    "religion": ["ክርስቲያን", "ሙስሊም", "ኦርቶዶክስ", "ፕሮቴስታንት", "ካቶሊክ"],
    "region": ["የአዲስ አበባ", "የጎንደር", "የመቀሌ", "የሀዋሳ", "የድሬዳዋ", "የጅማ", "የባህር ዳር"],
}


def swap_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for a, b in GENDER_PAIRS + PROFESSION_PAIRS:
        m[a], m[b] = b, a
    return m


def bias_terms() -> set[str]:
    return set(swap_map())

