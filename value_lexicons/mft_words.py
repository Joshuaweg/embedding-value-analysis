"""Moral Foundations Theory seed word lists for value embedding analysis."""

MFT_LEXICON: dict[str, dict[str, list[str]]] = {
    "care": {
        "virtue": [
            "care", "caring", "compassion", "compassionate", "empathy",
            "kindness", "gentle", "nurture", "protect", "shelter",
            "mercy", "merciful", "tender", "comfort", "heal",
            "sympathetic", "benevolent", "charitable",
        ],
        "vice": [
            "harm", "harmful", "hurt", "cruel", "cruelty",
            "abuse", "neglect", "suffer", "suffering", "damage",
            "destroy", "wound", "brutal", "ruthless", "savage",
            "torment", "pain", "violent",
        ],
    },
    "fairness": {
        "virtue": [
            "fair", "fairness", "equal", "equality", "just",
            "justice", "equitable", "reciprocal", "impartial", "balanced",
            "honest", "honesty", "rightful", "deserved", "merit",
            "reasonable", "proportional",
        ],
        "vice": [
            "cheat", "cheating", "betray", "betrayal", "unfair",
            "unjust", "injustice", "fraud", "deceive", "deception",
            "dishonest", "corrupt", "exploit", "biased", "prejudice",
            "discriminate", "unequal",
        ],
    },
    "loyalty": {
        "virtue": [
            "loyal", "loyalty", "faithful", "allegiance", "devoted",
            "devotion", "patriot", "patriotic", "solidarity", "comrade",
            "duty", "belonging", "united", "fellowship", "brotherhood",
            "sisterhood", "collective", "team",
        ],
        "vice": [
            "traitor", "treason", "betray", "disloyal", "deserter",
            "sedition", "apostate", "defect", "abandon", "forsake",
            "renegade", "unfaithful", "treacherous", "mutiny", "rebel",
            "secede", "heretic",
        ],
    },
    "authority": {
        "virtue": [
            "authority", "obey", "obedience", "respect", "order",
            "tradition", "hierarchy", "submit", "submission", "defer",
            "reverence", "duty", "discipline", "command", "sovereign",
            "rule", "law", "legitimate",
        ],
        "vice": [
            "rebel", "rebellion", "disobey", "defy", "defiance",
            "subvert", "subversion", "anarchy", "chaos", "overthrow",
            "usurp", "insubordinate", "lawless", "disrespect", "revolt",
            "mutiny", "unruly",
        ],
    },
    "sanctity": {
        "virtue": [
            "pure", "purity", "holy", "sacred", "sanctity",
            "divine", "blessed", "righteous", "pious", "wholesome",
            "clean", "chaste", "innocent", "consecrate", "reverent",
            "spiritual", "virtuous", "devout",
        ],
        "vice": [
            "corrupt", "corruption", "sin", "sinful", "taint",
            "defile", "profane", "desecrate", "impure", "filthy",
            "obscene", "degrade", "perverse", "wicked", "pollute",
            "blaspheme", "vile", "abomination",
        ],
    },
    "liberty": {
        "virtue": [
            "free", "freedom", "liberty", "autonomous", "autonomy",
            "sovereign", "independence", "independent", "emancipate", "liberate",
            "self-determination", "choice", "consent", "voluntary", "rights",
            "unshackled", "unfettered",
        ],
        "vice": [
            "oppress", "oppression", "tyranny", "tyrant", "enslave",
            "slavery", "coerce", "coercion", "dominate", "subjugate",
            "suppress", "constrain", "imprison", "dictator", "despotism",
            "authoritarian", "totalitarian",
        ],
    },
}

ALL_VALUE_WORDS: list[str] = sorted(set(
    word
    for foundation in MFT_LEXICON.values()
    for pole in foundation.values()
    for word in pole
))

WORD_TO_FOUNDATION: dict[str, str] = {
    word: foundation_name
    for foundation_name, poles in MFT_LEXICON.items()
    for pole in poles.values()
    for word in pole
}
