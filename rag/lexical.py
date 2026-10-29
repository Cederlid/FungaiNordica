"""BM25 keyword search over the stored chunks, with field-vocabulary expansion.

Users describe what they see in everyday English or Swedish ("under spruce",
"röd hatt"), while Funga Nordica uses Latin tree genera and its own
abbreviations ("Picea", "Sp deposit", "Frb"). `expand_query` adds the book's
terms so both the keyword and the embedding search can match them.
"""
import math
import re
from collections import Counter
from functools import lru_cache

# everyday term -> book term(s); keys are matched as whole words (lowercase)
_VOCAB = {
    # trees and hosts
    "spruce": "Picea", "fir": "Abies", "pine": "Pinus", "pines": "Pinus",
    "birch": "Betula", "birches": "Betula", "oak": "Quercus", "oaks": "Quercus",
    "beech": "Fagus", "aspen": "Populus", "poplar": "Populus", "willow": "Salix",
    "alder": "Alnus", "hazel": "Corylus", "larch": "Larix", "lime": "Tilia",
    "linden": "Tilia", "ash": "Fraxinus", "hornbeam": "Carpinus", "maple": "Acer",
    "elm": "Ulmus", "juniper": "Juniperus", "heather": "Calluna",
    # book abbreviations
    "spores": "sp", "spore": "sp", "print": "deposit",
    "basidiocarp": "frb", "fruitbody": "frb", "fruitbodies": "frb",
    # group names
    "milkcap": "Lactarius", "milkcaps": "Lactarius", "brittlegill": "Russula",
    "bolete": "tubes", "boletes": "tubes", "webcap": "Cortinarius", "bonnet": "Mycena",
    "waxcap": "Hygrocybe", "inkcap": "Coprinus", "ink": "deliquescent", "parasol": "Macrolepiota",
}

# Swedish -> book terms; only applied to Swedish queries ("tall", "ask", "bok" are English words too)
_VOCAB_SV = {
    "gran": "Picea", "granar": "Picea", "tall": "Pinus", "tallar": "Pinus",
    "björk": "Betula", "björkar": "Betula", "ek": "Quercus", "ekar": "Quercus",
    "bok": "Fagus", "asp": "Populus", "sälg": "Salix", "vide": "Salix",
    "al": "Alnus", "hassel": "Corylus", "lärk": "Larix", "lind": "Tilia",
    "ask": "Fraxinus", "avenbok": "Carpinus", "lönn": "Acer", "alm": "Ulmus",
    "ljung": "Calluna", "vitmossa": "Sphagnum", "barrskog": "coniferous",
    "lövskog": "deciduous",
    "svamp": "frb", "hatt": "cap", "hatten": "cap", "skivor": "gills", "skivorna": "gills",
    "fot": "stem", "foten": "stem", "fotbas": "stem base", "ring": "ring",
    "strumpa": "volva", "slida": "volva", "knöl": "bulb", "knölformad": "bulbous",
    "vårtor": "warts", "fjäll": "scales", "fjällig": "scaly", "rör": "tubes",
    "porer": "pores", "kött": "flesh", "köttet": "flesh", "mjölk": "milk",
    "sporer": "sp", "sporavtryck": "sp deposit", "sporpulver": "sp deposit",
    "lukt": "smell", "doft": "smell", "smak": "taste", "besk": "bitter",
    "bitter": "bitter", "skarp": "acrid", "brännande": "acrid", "mild": "mild",
    "slemmig": "viscid", "klibbig": "viscid", "sammetslen": "velvety",
    "röd": "red", "röda": "red", "vit": "white", "vita": "white", "gul": "yellow",
    "gula": "yellow", "brun": "brown", "bruna": "brown", "grå": "grey", "svart": "black",
    "grön": "green", "gröna": "green", "violett": "violet", "lila": "lilac",
    "rosa": "pink", "orange": "orange", "vinröd": "wine red", "blånar": "blue",
    "liten": "small", "små": "small", "stor": "large", "stora": "large",
    "tuvor": "clusters", "tuvad": "caespitose", "stubbar": "stumps", "stubbe": "stump",
    "ved": "wood", "mossa": "moss", "gräs": "grass", "höst": "autumn", "sommar": "summer",
    "giftig": "poisonous", "dödligt": "deadly", "ätlig": "edible",
    # Swedish group names
    "riska": "Lactarius", "riskor": "Lactarius", "kremla": "Russula", "kremlor": "Russula",
    "flugsvamp": "Amanita", "flugsvampar": "Amanita", "sopp": "Boletus", "soppar": "Boletus",
    "spindling": "Cortinarius", "spindlingar": "Cortinarius", "trådskivling": "Inocybe",
    "hätta": "Mycena", "hättor": "Mycena", "vaxskivling": "Hygrocybe", "musseron": "Tricholoma",
    "fjällskivling": "Lepiota", "kantarell": "Cantharellus", "champinjon": "Agaricus",
    "bläcksvamp": "Coprinus", "tofsskivling": "Pholiota", "slemskivling": "Hygrophorus",
    # habitats; values may be everyday English words, expanded further by _MORPH/_VOCAB
    "skog": "forest", "blandskog": "mixed forest", "blandad": "mixed", "löv": "deciduous leaves litter",
    "barr": "coniferous needles", "äng": "meadows", "hage": "pasture grassland", "bete": "pasture grazed",
    "mark": "soil", "gräsmatta": "lawns", "park": "parks", "trädgård": "gardens", "väg": "roadsides",
    "vägkant": "roadsides", "myr": "bogs", "mosse": "bogs Sphagnum", "kärr": "fens", "buske": "shrubs",
    "buskage": "shrubs", "gödsel": "dung manure", "dynga": "dung", "sand": "sandy", "kalk": "calcareous",
    "stam": "stem", "stjälk": "stem", "gälar": "gills", "lameller": "gills", "kant": "margin",
    "mitt": "centre", "vår": "spring", "ätbar": "edible",
    # surface and veil
    "flaga": "veil fragments", "flagor": "veil fragments", "prickar": "dots", "ränder": "striate",
    "randig": "striate", "strimmig": "striate", "strimmor": "striate", "räfflad": "sulcate striate",
    "fårad": "sulcate", "slät": "smooth glabrous", "glansig": "shiny", "matt": "dull",
    "rynkad": "wrinkled rugose", "spricker": "cracking", "sprucken": "cracked", "genomskinlig": "translucent striate",
    "torr": "dry", "fuktig": "moist", "blöt": "moist", "våt": "moist", "fibrig": "fibrillose",
    "fibrilig": "fibrillose", "trådig": "fibrillose", "skör": "brittle", "seg": "tough", "luden": "hairy",
    "hårig": "hairy", "ullig": "woolly", "tunn": "thin", "tjock": "thick", "ihålig": "hollow", "lång": "long",
    "rest": "remnants", "rester": "remnants", "slöja": "veil cortina", "spindelväv": "cortina",
    "vulst": "bulb", "säck": "volva", "nät": "net", "nätad": "net", "ådrad": "net",
    # gills and shape
    "tät": "crowded", "täta": "crowded", "trång": "crowded", "trånga": "crowded", "gles": "distant",
    "glesa": "distant", "nedlöpande": "decurrent", "nerlöpande": "decurrent", "utlöpande": "decurrent",
    "löper": "decurrent", "löpte": "decurrent", "fria": "free", "grenade": "forked", "gaffelgrenade": "forked",
    "ådror": "veins ridges", "åsar": "ridges", "tratt": "funnel", "trattformad": "funnel",
    "kägelformad": "conical", "konisk": "conical", "klockformad": "campanulate", "puckel": "umbo",
    "välvd": "convex", "platt": "plane", "nedsänkt": "depressed", "cylindrisk": "cylindrical",
    "äggformad": "ovoid", "oregelbunden": "irregular", "oregelbundna": "irregular", "klubbformad": "clavate",
    # smell, taste, colour changes
    "anis": "anise", "mjöl": "mealy", "mjölig": "mealy", "rättika": "radish", "gurka": "cucumber",
    "klor": "chlorine", "fruktig": "fruity", "honung": "honey", "mandel": "almond", "obehaglig": "unpleasant",
    "illa": "unpleasant", "kryddig": "spicy", "söt": "sweet", "jod": "iodine", "karbol": "carbolic",
    "bläck": "ink", "gulnar": "yellowing", "rodnar": "reddening", "svartnar": "blackening",
    "brunnar": "browning", "missfärgas": "discolouring", "flyter": "deliquescent", "upplöses": "deliquescent",
    # colours and colour parts of compounds ("blekgråblå", "olivgrön", "krämvit")
    "blek": "pale", "ljus": "pale", "mörk": "dark", "kräm": "cream", "elfenben": "ivory", "blå": "blue",
    "rött": "red", "grått": "grey", "ockra": "ochre", "rost": "rust", "purpur": "purple", "oliv": "olive",
    "tegel": "brick", "färg": "colour",
    # compound parts with no book term of their own
    "aktig": "", "färgad": "", "träd": "", "mönster": "", "häst": "", "stack": "", "saft": "", "gammal": "",
    "röra": "", "rört": "", "rörde": "",
}

# tried on words not in _VOCAB_SV, longest first: "granarna" -> "gran", "luktade" -> "lukt"
_SV_ENDINGS = ("arna", "erna", "orna", "ade", "en", "et", "na", "ar", "er", "or", "tt", "a", "e", "n", "t")

_SV_MARKERS = {"och", "med", "som", "hatt", "hatten", "skivor", "foten", "jag", "vilken", "eller", "svamp"}

_STOP = {
    "the", "a", "an", "is", "are", "and", "or", "of", "to", "in", "on", "with", "for",
    "from", "at", "by", "into", "how", "what", "when", "which", "it", "its", "this",
    "that", "does", "do", "not", "very", "has", "have", "be", "as", "i", "found",
    "mushroom", "mushrooms", "toadstool", "fungus", "fungi", "species", "growing",
    "grows", "grow", "under", "near", "like", "there", "my", "me",
    "och", "med", "som", "på", "under", "i", "en", "ett", "det", "är", "har", "vilken",
    "vilka", "jag", "hittade", "av", "till", "från", "eller", "inte", "mycket",
}

_TOKEN = re.compile(r"[a-zà-ÿ]+")


def _stem(t: str) -> str:
    # crude English plural folding so "gills"/"gill", "scales"/"scale" match
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


# field-guide wording -> the book's wording for the same character
_MORPH = {
    "wart": "veil fragments patches", "warts": "veil fragments patches", "warty": "veil fragments patches",
    "bulbous": "bulb", "bulb": "bulbous", "net": "reticulum reticulate", "netted": "reticulum reticulate",
    "network": "reticulum reticulate", "shaggy": "scaly fibrillose", "ink": "deliquescent",
    "inky": "deliquescent", "dissolves": "deliquescent", "dissolving": "deliquescent",
    "clustered": "caespitose fasciculate", "cluster": "caespitose fasciculate",
    "clusters": "caespitose fasciculate", "tufted": "caespitose fasciculate",
    "slimy": "viscid glutinous", "sticky": "viscid glutinous", "slippery": "viscid glutinous",
    "velvety": "tomentose", "woolly": "tomentose villose", "hairy": "tomentose villose",
    "pores": "tubes", "funnel": "infundibuliform", "radish": "raphanoid", "mealy": "farinaceous",
    "floury": "farinaceous", "peppery": "acrid", "hot": "acrid", "stumps": "wood", "roadsides": "roadsides",
    "roads": "roadsides", "conifers": "coniferous", "conifer": "coniferous", "lawns": "grass lawns",
    "meadows": "grassland meadows", "pointed": "umbo umbonate papilla", "umbo": "umbonate",
    "lawn": "grass lawns", "grass": "grassland lawns meadows", "grassy": "grassland lawns meadows",
    "park": "parks lawns grassland", "parks": "parks lawns grassland",
    "flakes": "veil fragments patches", "flaky": "veil fragments patches", "flecks": "veil fragments patches",
    "specks": "veil fragments patches", "patches": "veil fragments", "fuzzy": "tomentose",
    "spaced": "distant", "apart": "distant", "running": "decurrent", "runs": "decurrent",
    "fades": "hygrophanous", "faded": "hygrophanous", "fading": "hygrophanous", "drying": "hygrophanous",
    "bump": "umbo umbonate", "knob": "umbo umbonate", "strands": "rhizomorphs mycelial cords",
    "cords": "rhizomorphs mycelial cords", "skirt": "ring", "sac": "volva", "cup": "volva",
    "striations": "striate", "grooved": "sulcate striate", "wrinkled": "rugose", "beige": "buff",
    "bruises": "bruised", "bruising": "bruised", "aniseed": "anise", "forked": "forked dichotomous",
    "shiny": "glabrous", "fishy": "fishy", "scurfy": "furfuraceous",
}

# common English and Swedish names -> Funga Nordica names; matched as phrases
_NAMES = {
    "fly agaric": "Amanita muscaria", "panther cap": "Amanita pantherina", "death cap": "Amanita phalloides",
    "destroying angel": "Amanita virosa", "blusher": "Amanita rubescens", "tawny grisette": "Amanita fulva",
    "false death cap": "Amanita citrina", "penny bun": "Boletus edulis", "porcini": "Boletus edulis",
    "king bolete": "Boletus edulis", "bitter bolete": "Tylopilus felleus", "slippery jack": "Suillus luteus",
    "weeping bolete": "Suillus granulatus", "larch bolete": "Suillus grevillei",
    "orange birch bolete": "Leccinum versipelle", "brown birch bolete": "Leccinum scabrum",
    "false chanterelle": "Hygrophoropsis aurantiaca", "honey fungus": "Armillaria",
    "amethyst deceiver": "Laccaria amethystina", "the deceiver": "Laccaria laccata",
    "clouded agaric": "Clitocybe nebularis", "clouded funnel": "Clitocybe nebularis",
    "trooping funnel": "Infundibulicybe geotropa", "aniseed funnel": "Clitocybe odora",
    "fairy ring champignon": "Marasmius oreades", "velvet shank": "Flammulina velutipes",
    "oyster mushroom": "Pleurotus ostreatus", "shaggy ink cap": "Coprinus comatus",
    "lawyer's wig": "Coprinus comatus", "common ink cap": "Coprinopsis atramentaria",
    "sulphur tuft": "Hypholoma fasciculare", "brick tuft": "Hypholoma lateritium",
    "sheathed woodtuft": "Kuehneromyces mutabilis", "funeral bell": "Galerina marginata",
    "liberty cap": "Psilocybe semilanceata", "brown roll-rim": "Paxillus involutus",
    "poison pax": "Paxillus involutus", "deadly webcap": "Cortinarius rubellus",
    "fool's webcap": "Cortinarius orellanus", "yellow knight": "Tricholoma equestre",
    "man on horseback": "Tricholoma equestre", "soapy knight": "Tricholoma saponaceum",
    "sulphur knight": "Tricholoma sulphureum", "shaggy parasol": "Chlorophyllum rachodes",
    "stinking dapperling": "Lepiota cristata", "the sickener": "Russula emetica",
    "charcoal burner": "Russula cyanoxantha", "crab brittlegill": "Russula xerampelina",
    "ochre brittlegill": "Russula ochroleuca", "woolly milkcap": "Lactarius torminosus",
    "saffron milkcap": "Lactarius deliciosus", "false saffron milkcap": "Lactarius deterrimus",
    "rufous milkcap": "Lactarius rufus", "ugly milkcap": "Lactarius necator",
    "bleeding fairy helmet": "Mycena haematopus", "burgundydrop bonnet": "Mycena haematopus",
    "lilac bonnet": "Mycena pura", "blackening waxcap": "Hygrocybe conica",
    "parrot waxcap": "Hygrocybe psittacina", "snowy waxcap": "Hygrocybe virginea",
    "scarlet waxcap": "Hygrocybe coccinea", "plums and custard": "Tricholomopsis rutilans",
    "st george's mushroom": "Calocybe gambosa", "wood blewit": "Lepista nuda",
    "horse mushroom": "Agaricus arvensis", "field mushroom": "Agaricus campestris",
    "yellow stainer": "Agaricus xanthodermus", "deer shield": "Pluteus cervinus",
    "röd flugsvamp": "Amanita muscaria", "panterflugsvamp": "Amanita pantherina",
    "lömsk flugsvamp": "Amanita virosa", "grön flugsvamp": "Amanita phalloides",
    "vitgul flugsvamp": "Amanita citrina", "rodnande flugsvamp": "Amanita rubescens",
    "stensopp": "Boletus edulis", "karljohan": "Boletus edulis", "gallsopp": "Tylopilus felleus",
    "smörsopp": "Suillus luteus", "grynsopp": "Suillus granulatus", "lärksopp": "Suillus grevillei",
    "tegelsopp": "Leccinum versipelle", "björksopp": "Leccinum scabrum", "aspsopp": "Leccinum aurantiacum",
    "falsk kantarell": "Hygrophoropsis aurantiaca", "honungsskivling": "Armillaria",
    "ametistskivling": "Laccaria amethystina", "laxskivling": "Laccaria laccata",
    "pudrad trattskivling": "Clitocybe nebularis", "anistrattskivling": "Clitocybe odora",
    "nejlikbrosking": "Marasmius oreades", "vinterskivling": "Flammulina velutipes",
    "ostronmussling": "Pleurotus ostreatus", "fjällig bläcksvamp": "Coprinus comatus",
    "svavelgul slöjskivling": "Hypholoma fasciculare", "tegelröd slöjskivling": "Hypholoma lateritium",
    "föränderlig tofsskivling": "Kuehneromyces mutabilis", "gifthätting": "Galerina marginata",
    "toppslätskivling": "Psilocybe semilanceata", "pluggskivling": "Paxillus involutus",
    "toppig giftspindling": "Cortinarius rubellus", "orange giftspindling": "Cortinarius orellanus",
    "riddarmusseron": "Tricholoma equestre", "såpmusseron": "Tricholoma saponaceum",
    "svavelmusseron": "Tricholoma sulphureum", "stolt fjällskivling": "Macrolepiota procera",
    "giftkremla": "Russula emetica", "sillkremla": "Russula xerampelina",
    "skäggriska": "Lactarius torminosus", "granblodriska": "Lactarius deterrimus",
    "blodhätta": "Mycena haematopus", "rättikhätta": "Mycena pura",
    "toppig vaxskivling": "Hygrocybe conica", "papegojvaxskivling": "Hygrocybe psittacina",
    "snövit vaxskivling": "Hygrocybe virginea",
}


_BOOK_MIN_DF = 3  # book words this rare are often OCR noise, not correction targets
_BOOK_MIN_LEN = 7  # shorter words are one letter from too many book words ("fills" -> "fall")


def _edits(a: str, b: str) -> int:
    """Levenshtein distance, with adjacent swaps counted as one edit."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _closest(word: str, candidates, max_edits: int) -> str:
    near = [c for c in candidates if c[:1] == word[:1] and abs(len(c) - len(word)) <= max_edits]
    best = min(near, key=lambda c: _edits(word, c), default="")
    return best if best and _edits(word, best) <= max_edits else ""


def correct_spelling(query: str, vocab: dict) -> str:
    """Replace misspelled English words with the closest field-vocabulary or book word.

    `vocab` maps BM25 terms to the chunks containing them. Only words neither the
    book nor the vocabularies know are touched, at most one letter off (two in
    long words) and keeping the first letter; only long words are matched
    against the whole book: "worts" -> "warts", "unpleasent" ->
    "unpleasant". Swedish queries are left alone, their everyday words are not in
    the English book.
    """
    lower = query.lower()
    words = re.findall(r"[a-zà-ÿ]+", lower)
    if _is_swedish(lower, words):
        return query
    named = {w for name in _NAMES for w in name.split()}
    field = list(_MORPH) + list(_VOCAB)
    fixes = {}
    for w in set(words):
        if len(w) < 5 or w in _STOP or w in named or w in _MORPH or w in _VOCAB or _stem(w) in vocab:
            continue
        max_edits = 2 if len(w) >= 8 else 1
        fix = _closest(w, field, max_edits)
        if not fix and len(w) >= _BOOK_MIN_LEN:
            common = [t for t in vocab if t[:1] == w[:1] and len(vocab[t]) >= _BOOK_MIN_DF]
            fix = _closest(_stem(w), common, max_edits)
        if fix:
            fixes[w] = fix
    for w, fix in fixes.items():
        query = re.sub(rf"(?<![a-zà-ÿ]){w}(?![a-zà-ÿ])", fix, query, flags=re.IGNORECASE)
    return query


def _is_swedish(lower: str, words: list[str]) -> bool:
    return bool(re.search(r"[åäö]", lower)) or bool(_SV_MARKERS.intersection(words))


def _sv_base(w: str) -> str | None:
    if w in _VOCAB_SV:
        return w
    for e in _SV_ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 3 and w[:-len(e)] in _VOCAB_SV:
            return w[:-len(e)]
    return None


@lru_cache(maxsize=4096)
def _sv_terms(w: str) -> tuple[str, ...] | None:
    """Terms for a Swedish word read as a known word, an inflected form or a compound.

    A compound is split only when every part is known ("granskogen" -> gran + skog,
    "gulbrun" -> gul + brun, "betesmark" -> bete(s) + mark), so "bokstav" stays unread.
    """
    base = _sv_base(w)
    if base:
        return (_VOCAB_SV[base],)
    for i in range(len(w) - 2, 1, -1):
        head, tail = w[:i], w[i:]
        if head not in _VOCAB_SV and not (head.endswith("s") and head[:-1] in _VOCAB_SV):
            continue
        rest = _sv_terms(tail)
        if rest is not None:
            return (_VOCAB_SV[head] if head in _VOCAB_SV else _VOCAB_SV[head[:-1]],) + rest
    return None


def expand_query(query: str) -> str:
    """Append the book's terms for everyday/Swedish words found in the query."""
    lower = query.lower()
    words = re.findall(r"[a-zà-ÿ']+", lower)
    swedish = _is_swedish(lower, words)
    # longest names first, so "false death cap" is not also read as "death cap"
    extra, rest = [], lower
    for name in sorted(_NAMES, key=len, reverse=True):
        if re.search(rf"(?<![a-zà-ÿ]){re.escape(name)}(?![a-zà-ÿ])", rest):
            extra.append(_NAMES[name])
            rest = rest.replace(name, " ")
    for w in words:
        if w in _MORPH:
            extra.append(_MORPH[w])
        if w in _VOCAB:
            extra.append(_VOCAB[w])
        elif swedish:
            for term in _sv_terms(w) or ():
                extra.append(term)
                # "nät" -> "net" -> "reticulum reticulate"
                extra.extend(_MORPH.get(t) or _VOCAB.get(t) or "" for t in term.split())
    if "spore print" in query.lower():
        extra.append("sp deposit")
    extra = [e for e in dict.fromkeys(extra) if e and e.lower() not in query.lower()]
    return f"{query} ({', '.join(extra)})" if extra else query


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.index = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.index.setdefault(t, []).append(i)

    def top(self, query: str, k: int) -> list[int]:
        scores = Counter()
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.index[t]:
                f = self.tfs[i][t]
                norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        return [i for i, _ in scores.most_common(k)]
