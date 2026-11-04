"""Deterministic red-flag rules for Asklepios (human triage).

A SAFETY FLOOR under the AI nurse, never a replacement for it. The rules read what the USER wrote (not the AI's
reply), and they can only RAISE urgency. Matching is done on lower-cased, accent-stripped text with final sigma
folded to sigma, so Greek and English both work. Write patterns WITHOUT accents and with a plain sigma; Greek
characters inside a pattern are folded automatically by _fold_pat().

Sources are published guidelines found in PubMed (PMIDs checked against the PubMed record). The rule wording and
the choice of which statements are "act now" are OUR reading of those guidelines: every rule is basis="policy" and
needs review by a licensed clinician before the label can be removed.

Kill switch: env TRIAGE_FLOOR=off disables the rules. The AI-failure fallback stays on.
Levels use the app's own scale: EMERGENCY > URGENT > SELF_CARE.
"""
import os
import re
import unicodedata
from dataclasses import dataclass, field

RANK = {"SELF_CARE": 0, "URGENT": 1, "EMERGENCY": 2}


def _pm(pmid):
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


# PMIDs verified against the PubMed record (title, journal, year) on 2026-10-25.
SRC = {
    "acs": _pm("37622654"),        # 2023 ESC Guidelines for the management of acute coronary syndromes
    "pe": _pm("31504429"),         # 2019 ESC Guidelines, acute pulmonary embolism
    "aorta": _pm("25173340"),      # 2014 ESC Guidelines, aortic diseases
    "stroke": _pm("31662037"),     # AHA/ASA 2019 early management of acute ischemic stroke
    "fast": _pm("12511753"),       # Face Arm Speech Test, Stroke 2003 (recognition instrument)
    "bls": _pm("33773835"),        # ERC Guidelines 2021: Basic Life Support
    "first_aid": _pm("33773828"),  # ERC Guidelines 2021: First aid
    "als": _pm("33773825"),        # ERC Guidelines 2021: Adult advanced life support
    "ana_wao": _pm("33204386"),    # World Allergy Organization anaphylaxis guidance 2020
    "ana_eaaci": _pm("34343358"),  # EAACI guidelines: Anaphylaxis (2021 update)
    "meningitis": _pm("27062097"),  # ESCMID guideline: acute bacterial meningitis
    "sah": _pm("22556195"),        # AHA/ASA 2012 aneurysmal subarachnoid hemorrhage
    "head": _pm("23432764"),       # Scandinavian guidelines for mild/moderate head injuries in adults
    "syncope": _pm("29562304"),    # 2018 ESC Guidelines, syncope
    "dka": _pm("39052901"),        # Hyperglycemic crises in adults with diabetes: consensus report
    "seizure": _pm("26900382"),    # AES guideline: convulsive status epilepticus (seizure > 5 min)
    "infant": _pm("34281996"),     # AAP guideline: febrile infants 8 to 60 days
}
POLICY = ""  # no published source found yet: needs a clinician


def norm(text):
    if not text:
        return ""
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("ς", "σ")


def _fold_pat(p):
    return "".join(norm(c) if "Ͱ" <= c <= "Ͽ" else c for c in p)


@dataclass
class Rule:
    id: str
    level: str
    all_of: tuple
    en: str
    el: str
    source: str = ""
    basis: str = "policy"
    exclude: str = ""
    neg_guard: bool = False


@dataclass
class Hit:
    id: str
    level: str
    en: str
    el: str
    source: str
    basis: str


@dataclass
class Result:
    level: str = ""
    hits: list = field(default_factory=list)


def _r(*patterns):
    return tuple(re.compile(_fold_pat(p), re.S) for p in patterns)


# ---- shared fragments (Greek written with accents is folded by _fold_pat) -----------------------------------------
_CHEST = (r"chest (pain|pressure|tight\w*|heav\w*|discomfort)|pain (in|across) (my |the )?chest|pressure (in|on) (my |the )?chest|"
          r"(πον|πιεσ|σφιξ|βαρ|ματσ|καψιμ)\w* (στο|στον|στα|στη|στην) (στηθ|θωρακ)|πονω στο στηθος|στηθος μου (πονα|σφιγγ)")
_BREATHLESS = (r"short(ness)? of breath|can'?t (get my )?breath|cannot breathe|can'?t breathe|struggling to breathe|gasping|"
               r"difficulty breathing|trouble breathing|breathless|"
               r"δυσπνο|δυσκολια (στην )?αναπνο|δεν μπορω να αναπνευσω|δεν παιρνω αναπνοη|λαχανιαζ|ασφυξι|"
               r"κοβεται η αναπνοη|παλευω να αναπνευσω")
_NEG_WORDS = r"(no|not|without|never|denies|denied|δεν|οχι|χωρις|ουτε)"

RULES = [
    # ------------------------------------------------------------------ heart / chest
    Rule("CHEST-ACS", "EMERGENCY",
         _r(_CHEST,
            r"arm|jaw|neck|back|shoulder|sweat|cold sweat|nause|vomit|faint|dizz|short(ness)? of breath|breathless|"
            r"βραχιον|χερι|σαγονι|λαιμο|πλατη|ωμο|ιδρωτ|ναυτι|εμετ|λιποθυμ|ζαλαδ|δυσπνο|λαχανιαζ"),
         "Chest pain with pain spreading, sweating, nausea or breathlessness can be a heart attack. Call 166 (EKAB) or 112 now.",
         "Πόνος στο στήθος με επέκταση του πόνου, ιδρώτα, ναυτία ή δύσπνοια μπορεί να είναι έμφραγμα. Κάλεσε τώρα 166 (ΕΚΑΒ) ή 112.",
         SRC["acs"]),
    Rule("CHEST-ANY", "URGENT",
         _r(_CHEST),
         "New chest pain or pressure needs a medical check today. Call 166 or 112 if it is severe, lasts more than a few minutes or comes with other symptoms.",
         "Νέος πόνος ή πίεση στο στήθος χρειάζεται ιατρικό έλεγχο σήμερα. Κάλεσε 166 ή 112 αν είναι έντονος, κρατάει πάνω από λίγα λεπτά ή συνοδεύεται από άλλα συμπτώματα.",
         SRC["acs"], neg_guard=True),
    Rule("AORTA-TEARING", "EMERGENCY",
         _r(r"tearing|ripping|stabbing|knife|worst pain|sudden (severe|sharp)|σκισιμο|σκιζ|σχιζ|μαχαιρι|χειροτερος πονος|ξαφνικος (εντονος|οξυς)",
            r"chest|back|between (my )?shoulder|στηθ|θωρακ|πλατη|ραχη"),
         "A sudden tearing pain in the chest or back can be an aortic emergency. Call 166 or 112 now.",
         "Ξαφνικός σκίζων πόνος στο στήθος ή την πλάτη μπορεί να είναι επείγον πρόβλημα της αορτής. Κάλεσε τώρα 166 ή 112.",
         SRC["aorta"]),
    Rule("PE-SUDDEN", "EMERGENCY",
         _r(r"sudden|suddenly|all of a sudden|ξαφνικ|απροσδοκητ", _BREATHLESS,
            r"calf|leg swell|swollen leg|long flight|long (car )?journey|surgery|operation|immobil|bed ?rest|cancer|clot|thrombo|pregnan|"
            r"γαμπα|πρησμεν\w* (ποδι|ποδια)|μακρινη πτηση|μακρυ ταξιδι|εγχειρηση|ακινητ|καρκιν|θρομβ|εγκυ"),
         "Sudden breathlessness with a swollen calf, recent surgery, a long journey or a clot history can be a blood clot in the lungs. Call 166 or 112 now.",
         "Ξαφνική δύσπνοια με πρησμένη γάμπα, πρόσφατη επέμβαση, μακρύ ταξίδι ή ιστορικό θρόμβου μπορεί να είναι πνευμονική εμβολή. Κάλεσε τώρα 166 ή 112.",
         SRC["pe"]),
    Rule("SYNCOPE-HEART", "URGENT",
         _r(r"faint|passed out|blacked out|lost consciousness|λιποθυμ|εχασα τις αισθησεις|εκανα λιποθυμια",
            r"palpitat|racing heart|heart (racing|pounding|skipp)|irregular|chest|exercise|exertion|running|παλμ|ταχυκαρδ|ακανονιστ|στηθ|ασκηση|τρεξιμο|προσπαθεια"),
         "Fainting with palpitations, chest symptoms or on exertion needs a medical check the same day.",
         "Λιποθυμία με αίσθημα παλμών, συμπτώματα στο στήθος ή κατά την προσπάθεια χρειάζεται ιατρικό έλεγχο την ίδια μέρα.",
         SRC["syncope"]),
    # ------------------------------------------------------------------ brain / nerves
    Rule("STROKE-FAST", "EMERGENCY",
         _r(r"face (droop|dropp|is droop|numb)|droop(ing|y)? (face|mouth)|mouth (droop|dropp|crooked)|crooked (face|smile|mouth)|slurr|can'?t (speak|talk|find (the )?words)|"
            r"(speech|speaking|talking) (is )?(slurred|garbled|strange|difficult)|trouble (speaking|talking)|"
            r"(one|left|right) (arm|leg|side).{0,20}(weak|numb|heavy|paraly|can'?t (move|lift))|sudden(ly)? .{0,25}(weak|numb|paraly)\w*|(weak|numb)\w* (on|down) (one|the (left|right)) side|"
            r"sudden (loss of vision|blind|double vision|confusion)|"
            r"στραβ\w* (στομα|προσωπο|χαμογελο)|πεφτει (το )?(στομα|προσωπο)|(προσωπο|στομα) .{0,15}(πεφτ|στραβ|μουδιασ)|"
            r"μπερδευ\w* (τα )?λογια|δεν μπορει να μιλησ|δυσκολια στην ομιλια|θολη ομιλια|"
            r"(ενα|αριστερ\w*|δεξι\w*) (χερι|ποδι|πλευρα).{0,25}(αδυναμ|μουδιασ|παραλυ|βαρι|δεν (κουνιεται|σηκωνεται))|ξαφνικ\w* .{0,25}(αδυναμ|μουδιασ|παραλυ)\w*|"
            r"ξαφνικη (απωλεια ορασης|τυφλωση|διπλοπια|συγχυση)"),
         "A drooping face, slurred speech or sudden weakness or numbness of one arm or leg can be a stroke. Call 166 or 112 now and note the time it started.",
         "Πεσμένο πρόσωπο, μπερδεμένη ομιλία ή ξαφνική αδυναμία ή μούδιασμα στο χέρι ή το πόδι μπορεί να είναι εγκεφαλικό. Κάλεσε τώρα 166 ή 112 και σημείωσε την ώρα που ξεκίνησε.",
         SRC["stroke"], neg_guard=True),
    Rule("HEADACHE-THUNDER", "EMERGENCY",
         _r(r"worst headache|thunderclap|sudden (severe|terrible|explosive) headache|headache.{0,25}(sudden|worst|explod)|"
            r"χειροτερος πονοκεφαλος|ξαφνικος (εντονος|τρομερος) πονοκεφαλος|πονοκεφαλος.{0,25}(ξαφνικ|χειροτερ|εκρηξ)|σαν χτυπημα στο κεφαλι"),
         "A sudden, very severe headache ('the worst of my life') can mean bleeding in the brain. Call 166 or 112 now.",
         "Ένας ξαφνικός, πολύ έντονος πονοκέφαλος («ο χειρότερος της ζωής μου») μπορεί να σημαίνει αιμορραγία στον εγκέφαλο. Κάλεσε τώρα 166 ή 112.",
         SRC["sah"]),
    Rule("MENINGITIS", "EMERGENCY",
         _r(r"fever|high temperature|πυρετ|θερμοκρασια",
            r"stiff neck|neck (is )?stiff|can'?t (bend|touch) (my )?(chin|neck)|rash that (does ?n'?t|won'?t) (fade|go)|non.?blanching|purple (rash|spots)|"
            r"light (hurts|sensitiv)|photophobia|σκληρος αυχενας|ακαμψια αυχεν|δεν σκυβει ο λαιμος|εξανθημα που δεν (φευγει|ξεθωριαζει)|μωβ (κηλιδες|εξανθημα)|φωτοφοβια"),
         "Fever with a stiff neck, a rash that does not fade under pressure or light sensitivity can be meningitis. Call 166 or 112 now.",
         "Πυρετός με σκληρό αυχένα, εξάνθημα που δεν ξεθωριάζει με πίεση ή φωτοφοβία μπορεί να είναι μηνιγγίτιδα. Κάλεσε τώρα 166 ή 112.",
         SRC["meningitis"]),
    Rule("SEIZURE-LONG", "EMERGENCY",
         _r(r"seizure|convuls|fitting|epilep|σπασμ|επιληπ|κριση επιληψιας",
            r"(more than|over|for|lasting|lasted) .{0,10}(five|5|ten|10|fifteen|15|20|twenty) ?(min|λεπτ)|won'?t stop|not stopping|still (seizing|fitting|convulsing)|"
            r"again and again|one after another|second seizure|not waking|first (ever )?seizure|"
            r"πανω απο (πεντε|5|δεκα|10) λεπτ|δεν σταματ|ακομα (σπαζ|κανει σπασμ)|ξανα και ξανα|διαδοχικ|πρωτη (φορα )?κριση|δεν ξυπνα"),
         "A seizure lasting over 5 minutes, repeating without recovery, or a first-ever seizure needs emergency care. Call 166 or 112 now.",
         "Σπασμοί που διαρκούν πάνω από 5 λεπτά, επαναλαμβάνονται χωρίς ανάρρωση ή είναι η πρώτη φορά χρειάζονται επείγουσα φροντίδα. Κάλεσε τώρα 166 ή 112.",
         SRC["seizure"]),
    Rule("SEIZURE-ANY", "URGENT",
         _r(r"seizure|convuls|epileptic fit|σπασμ|επιληπτικη κριση"),
         "A seizure should be checked by a doctor the same day.",
         "Μια επιληπτική κρίση χρειάζεται έλεγχο από γιατρό την ίδια μέρα.",
         SRC["seizure"], neg_guard=True),
    Rule("HEAD-INJURY", "EMERGENCY",
         _r(r"head injur|hit (my |his |her |the )?head|bang(ed)? (my |his |her |the )?head|fell on (my |his |her |the )?head|blow to the head|"
            r"χτυπησ\w* (το )?κεφαλι|χτυπημα στο κεφαλι|κτυπησ\w* (το )?κεφαλι|ετριξε το κεφαλι|επεσε στο κεφαλι|τραυμα στο κεφαλι",
            r"vomit|threw up|confus|drows|can'?t stay awake|unconscious|lost consciousness|blacked out|seizure|blood (from|out of) (the )?(ear|nose)|clear fluid|"
            r"worsening headache|weak|slurr|double vision|"
            r"εμετ|εκανε εμετο|σαστισμ|συγχυση|νυστα|υπνηλι|δεν μπορει να μεινει ξυπνιος|εχασε τις αισθησεις|λιποθυμ|σπασμ|αιμα απο (το )?(αυτι|μυτη)|διαυγες υγρο|πονοκεφαλος που χειροτερευ|αδυναμι|διπλοπια"),
         "A head injury with vomiting, confusion, drowsiness, a seizure or blood or clear fluid from the ear or nose needs emergency care. Call 166 or 112 now.",
         "Χτύπημα στο κεφάλι με εμετό, σύγχυση, υπνηλία, σπασμό ή αίμα ή διαυγές υγρό από αυτί ή μύτη χρειάζεται επείγουσα φροντίδα. Κάλεσε τώρα 166 ή 112.",
         SRC["head"], neg_guard=True),
    # ------------------------------------------------------------------ breathing / allergy / consciousness / bleeding
    Rule("BREATH-SEVERE", "EMERGENCY",
         _r(r"can'?t breathe|cannot breathe|can'?t get (my )?breath|gasping|struggling to breathe|blue lips|lips (are |turning )?blue|turning blue|"
            r"can'?t (speak|talk) (a )?(full )?(sentence|more than)|"
            r"δεν μπορω να αναπνευσω|δεν παιρνω αναπνοη|δεν αναπνε[ιυ]|ασφυξι|μπλε (χειλη|στομα)|χειλη .{0,10}μπλε|μελανιαζ|"
            r"δεν μπορει να μιλησει .{0,10}απο την δυσπνοια|παλευει να αναπνευσει"),
         "Severe difficulty breathing, blue lips or being unable to speak is an emergency. Call 166 or 112 now.",
         "Σοβαρή δυσκολία στην αναπνοή, μπλε χείλη ή αδυναμία ομιλίας είναι επείγον. Κάλεσε τώρα 166 ή 112.",
         SRC["bls"], neg_guard=True),
    Rule("ANAPHYLAXIS", "EMERGENCY",
         _r(r"swell\w* (of )?(the |my |his |her )?(lips?|tongue|throat|face|mouth)|(lips?|tongue|throat|face).{0,12}swo?ll?\w*|throat (is )?(closing|tight)|hives|rash all over|"
            r"πρησμεν\w* (χειλ|γλωσσ|λαιμ|προσωπ|στομα)|πρηστηκ\w* (τα |η |ο |το )?(χειλ|γλωσσ|λαιμ|προσωπ)|πρηζ\w* (το |η |ο |τα )?(χειλ|γλωσσ|λαιμ|προσωπ)|κλεινει ο λαιμος|φαγουρα παντου|κνιδωση|εξανθημα παντου",
            r"sting|bite|ate|eaten|medicine|medication|antibiotic|injection|peanut|nut|shellfish|allerg|"
            r"τσιμπημα|δαγκωμα|εφαγε|φαγητο|φαρμακο|αντιβιοτικ|ενεση|φυστικ|ξηροι καρποι|οστρακο|αλλεργ",
            r"breath|wheez|dizz|faint|collaps|blood pressure|throat|swallow|αναπνο|συριγμ|ζαλαδ|λιποθυμ|καταρρευσ|πιεση|καταποση|λαιμ"),
         "Swelling of the lips, tongue or throat with breathing trouble, dizziness or fainting after a sting, food or medicine can be anaphylaxis. Call 166 or 112 now; use an adrenaline auto-injector if you have one.",
         "Πρήξιμο στα χείλη, τη γλώσσα ή το λαιμό με δυσκολία στην αναπνοή, ζάλη ή λιποθυμία μετά από τσίμπημα, φαγητό ή φάρμακο μπορεί να είναι αναφυλαξία. Κάλεσε τώρα 166 ή 112· αν έχεις ενέσιμη αδρεναλίνη, χρησιμοποίησέ την.",
         SRC["ana_wao"]),
    Rule("UNRESPONSIVE", "EMERGENCY",
         _r(r"unconscious|unresponsive|not responding|won'?t wake|can'?t (be )?wake|not breathing|stopped breathing|collapsed|no pulse|"
            r"αναισθητ|δεν αντιδρα|δεν ξυπνα|δεν μπορω να τον ξυπνησω|δεν αναπνε[ιυ]|σταματησε να αναπνεει|καταρρευσε|δεν εχει σφυξεις"),
         "A person who does not respond or is not breathing normally needs emergency help. Call 166 or 112 now and start chest compressions if they are not breathing.",
         "Άτομο που δεν ανταποκρίνεται ή δεν αναπνέει φυσιολογικά χρειάζεται άμεση βοήθεια. Κάλεσε τώρα 166 ή 112 και ξεκίνα θωρακικές συμπιέσεις αν δεν αναπνέει.",
         SRC["bls"], neg_guard=True),
    Rule("BLEEDING-SEVERE", "EMERGENCY",
         _r(r"won'?t stop bleeding|(bleeding|blood) .{0,20}(won'?t stop|not stopping|soaking|spurting|gushing|pouring|a lot)|"
            r"(spurting|gushing|pouring) blood|vomit\w* blood|coughing (up )?blood|blood in (my )?vomit|black (tarry )?stool|tarry stool|"
            r"δεν σταματα η αιμορραγια|(αιμορραγια|αιμα) .{0,20}(δεν σταματα|τρεχει|πετατ|ποτιζ|πολυ)|ξερναω αιμα|ξερασμα με αιμα|βηχω αιμα|μαυρα κοπρανα|μελανα κοπρανα"),
         "Bleeding that will not stop, vomiting or coughing blood, or black tarry stool needs emergency care. Press firmly on a visible wound and call 166 or 112 now.",
         "Αιμορραγία που δεν σταματά, εμετός ή βήχας με αίμα ή μαύρα κοπρανά χρειάζονται επείγουσα φροντίδα. Πίεσε σταθερά την πληγή και κάλεσε τώρα 166 ή 112.",
         SRC["first_aid"], neg_guard=True),
    # ------------------------------------------------------------------ mental health / poisoning
    Rule("SUICIDE", "EMERGENCY",
         _r(r"kill (my|him|her)self|end (my|his|her) life|want(ed)? to die|wish i (was|were) dead|suicid|take my (own )?life|no reason to live|better off dead|"
            r"θελω να πεθανω|θα αυτοκτονησω|να αυτοκτονησω|αυτοκτον|να τελειωσω τη ζωη μου|να σκοτωσω τον εαυτο μου|δεν αξιζει να ζω|καλυτερα να ημουν νεκρος"),
         "You do not have to be alone with this. Please call 112 now or go to the nearest emergency department, and tell someone you trust where you are.",
         "Δεν χρειάζεται να το περάσεις μόνος/η. Κάλεσε τώρα το 112 ή πήγαινε στο πλησιέστερο τμήμα επειγόντων και πες σε κάποιον που εμπιστεύεσαι πού βρίσκεσαι.",
         "https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/behaviours/help-for-suicidal-thoughts/"),
    Rule("OVERDOSE", "EMERGENCY",
         _r(r"overdos|took (too many|a whole|all the)|swallowed (a (lot|bottle)|bleach|pills|tablets|poison|battery)|drank (bleach|poison|antifreeze|cleaner)|poison(ed|ing)|"
            r"υπερδοσ|πηρε (πολλα|ολα τα) (χαπια|δισκια)|καταπι\w* (απορρυπαντικ|χλωρ|χαπια|δισκια|δηλητηρι|μπαταρι)|ηπιε (χλωρ|αντιψυκτικ|δηλητηρι)|δηλητηριασ"),
         "Taking too much medicine or swallowing a poison or chemical needs emergency advice. Call 166 or 112 now and keep the package. Do not make them vomit.",
         "Η λήψη υπερβολικής ποσότητας φαρμάκου ή η κατάποση δηλητηρίου ή χημικού χρειάζεται άμεση συμβουλή. Κάλεσε τώρα 166 ή 112 και κράτα τη συσκευασία. Μην προκαλέσεις εμετό.",
         "https://www.nhs.uk/conditions/poisoning/", neg_guard=True),
    # ------------------------------------------------------------------ children / pregnancy / other
    Rule("INFANT-FEVER", "EMERGENCY",
         _r(r"newborn|new born|\b([1-9]|[1-5]\d|60) days?[- ]old|\b[1-9] weeks?[- ]old|\b[1-2] months?[- ]old|νεογεν|\b([1-9]|[1-5]\d|60) (ημερων|ημερα)|\b[1-9] εβδομαδ\w*|\b[1-2] μην\w*",
            r"fever|temperature (of )?(38|39|40)|38[.,]\d|39[.,]\d|πυρετ"),
         "A fever in a baby under about 2 months old needs to be seen by a doctor straight away. Go to an emergency department now or call 166 or 112.",
         "Πυρετός σε μωρό κάτω από περίπου 2 μηνών πρέπει να εξεταστεί αμέσως από γιατρό. Πήγαινε τώρα σε τμήμα επειγόντων ή κάλεσε 166 ή 112.",
         SRC["infant"]),
    Rule("PREGNANCY-DANGER", "EMERGENCY",
         _r(r"pregnan|εγκυ|εγκυμοσυνη|κυηση",
            r"heavy bleeding|bleeding|severe (abdominal |stomach )?pain|no (fetal )?movement|stopped moving|baby (is )?not moving|waters broke|seizure|severe headache.{0,20}(vision|swelling)|"
            r"αιμορραγι|εντονος πονος|δεν κινειται το μωρο|δεν νιωθω κινησεις|εσπασε ο υδρωπας|εσπασαν τα νερα|σπασμ|εντονος πονοκεφαλος.{0,20}(ορασ|πρηξιμο)"),
         "Bleeding, severe pain, no baby movements, waters breaking or a seizure in pregnancy needs urgent care. Call 166 or 112 or go to the maternity emergency department now.",
         "Αιμορραγία, έντονος πόνος, απουσία κινήσεων του μωρού, ρήξη υμένων ή σπασμός στην εγκυμοσύνη χρειάζεται επείγουσα φροντίδα. Κάλεσε 166 ή 112 ή πήγαινε τώρα στα επείγοντα μαιευτηρίου.",
         "https://www.nhs.uk/pregnancy/related-conditions/common-symptoms/vaginal-bleeding/"),
    Rule("DIABETES-CRISIS", "EMERGENCY",
         _r(r"diabet|sugar|insulin|διαβητ|ζαχαρο|ινσουλιν",
            r"confus|unresponsive|won'?t wake|seizure|fruity breath|deep (fast )?breath|vomit\w* .{0,20}(thirst|drink)|very (thirsty|drowsy)|"
            r"συγχυση|δεν ξυπνα|σπασμ|μυρωδια φρουτ|βαθιες αναπνοες|εμετ\w* .{0,20}(διψα)|πολυ (διψα|υπνηλι)"),
         "A person with diabetes who is confused, very drowsy, vomiting or breathing deeply, or who cannot be woken, needs emergency care. Call 166 or 112 now.",
         "Άτομο με διαβήτη που είναι σε σύγχυση, πολύ νυσταγμένο, κάνει εμετούς ή αναπνέει βαθιά ή δεν ξυπνά χρειάζεται επείγουσα φροντίδα. Κάλεσε τώρα 166 ή 112.",
         SRC["dka"]),
    Rule("TESTICLE-PAIN", "EMERGENCY",
         _r(r"testic|testes|scrot|ορχ|οσχε",
            r"sudden|suddenly|severe|swollen|high|twist|ξαφνικ|εντονος|πρησμεν|ανεβηκε|στριψ"),
         "Sudden, severe pain or swelling in a testicle can be a twisted testicle, where minutes matter. Go to an emergency department now or call 166 or 112.",
         "Ξαφνικός, έντονος πόνος ή πρήξιμο στον όρχι μπορεί να είναι στρέψη όρχεως, όπου μετράνε τα λεπτά. Πήγαινε τώρα σε τμήμα επειγόντων ή κάλεσε 166 ή 112.",
         "https://medlineplus.gov/ency/article/000517.htm"),
    Rule("EYE-CHEMICAL", "EMERGENCY",
         _r(r"(chemical|bleach|acid|cleaner|oven cleaner|lime|χημικ|χλωρ|οξυ|καθαριστικ).{0,25}(in|into|got in|splash\w*|στο|στα) .{0,10}(eye|ματι|ματια)|"
            r"(eye|ματι|ματια).{0,25}(chemical|bleach|acid|cleaner|χημικ|χλωρ|οξυ|καθαριστικ)"),
         "A chemical in the eye is an emergency. Rinse the eye with clean running water for at least 15 minutes and call 112 or go to an emergency department.",
         "Χημική ουσία στο μάτι είναι επείγον. Ξέπλυνε το μάτι με καθαρό τρεχούμενο νερό για τουλάχιστον 15 λεπτά και κάλεσε 112 ή πήγαινε σε τμήμα επειγόντων.",
         "https://medlineplus.gov/ency/article/000054.htm"),
]

# Clear statements that justify skipping the question cycle and showing a fixed emergency message. Hand-picked;
# the other EMERGENCY rules still raise the final level. Needs clinician sign-off like everything else.
ACT_NOW = frozenset({
    "CHEST-ACS", "STROKE-FAST", "BREATH-SEVERE", "ANAPHYLAXIS", "UNRESPONSIVE", "BLEEDING-SEVERE", "SUICIDE",
    "OVERDOSE", "SEIZURE-LONG", "AORTA-TEARING", "HEADACHE-THUNDER", "MENINGITIS", "INFANT-FEVER",
})

_NEG = re.compile(
    _fold_pat(r"(\bno\b|\bnot\b|\bnever\b|\bwithout\b|\bdenies\b|\bdidn'?t\b|\bdoesn'?t\b|\bhasn'?t\b|\bhaven'?t\b|"
              r"δεν|οχι|χωρις|ουτε|ποτε)[^.;\n]{0,25}(chest pain|pain|breath|bleed|seizure|faint|unconscious|swell|hives|"
              r"stroke|weak|numb|overdos|poison|suicid|hit|head|vomit|"
              r"πονο|αναπνο|αιμορραγι|σπασμ|λιποθυμ|πρηξιμο|εγκεφαλικ|αδυναμι|μουδιασ|υπερδοσ|δηλητηρι|αυτοκτον|χτυπ|κεφαλι|εμετ)"),
    re.S)
_UNSURE = re.compile(_fold_pat(r"maybe|might|not sure|unsure|i think|perhaps|possibly|ισως|μηπως|δεν ειμαι σιγουρ|νομιζω|πιθανον"), re.S)


def strip_negated(text):
    """Remove clauses where the user says the symptom is NOT there. Uncertain wording is kept (safer)."""
    def repl(m):
        window = text[max(0, m.start() - 20):m.end()]
        return " " if not _UNSURE.search(window) else m.group(0)
    return _NEG.sub(repl, text)


def enabled():
    return os.environ.get("TRIAGE_FLOOR", "on").strip().lower() not in ("off", "0", "false", "no")


def evaluate(user_texts):
    """Check every USER message of the current chat (never the AI's replies). Returns Result."""
    if not enabled():
        return Result()
    text = norm(" \n ".join(t for t in (user_texts or []) if t))
    if not text.strip():
        return Result()
    tg = strip_negated(text)
    hits = []
    for r in RULES:
        t = tg if r.neg_guard else text
        if r.exclude and re.search(_fold_pat(r.exclude), t):
            continue
        if all(p.search(t) for p in r.all_of):
            hits.append(Hit(r.id, r.level, r.en, r.el, r.source, r.basis))
    res = Result(hits=hits)
    if hits:
        res.level = max((h.level for h in hits), key=lambda x: RANK[x])
    return res


def act_now_hits(res):
    return [h for h in res.hits if h.id in ACT_NOW]


def rule_key(res):
    """Stable key of the act-now rule set, so the fixed message is shown once per set."""
    return ",".join(sorted(h.id for h in act_now_hits(res)))


def act_now_reply(res, lang="en"):
    """Fixed emergency message built from the rules that fired. No AI call."""
    el = lang == "el"
    hs = act_now_hits(res)
    head = "🚨 **Κάλεσε τώρα 166 (ΕΚΑΒ) ή 112.**" if el else "🚨 **Call 166 (EKAB) or 112 now.**"
    lines = [h.el if el else h.en for h in hs]
    tail = ("Δεν χρειάζεται να απαντήσεις σε άλλες ερωτήσεις. Μείνε με κάποιον αν γίνεται και πες τους τι συμβαίνει. "
            "Η εφαρμογή δεν διαγνώσκει· αυτό το μήνυμα βασίζεται σε κανόνες ασφαλείας που αναμένουν ακόμη έλεγχο από κλινικό ιατρό."
            if el else
            "You do not need to answer more questions. Stay with someone if you can and tell them what is happening. "
            "This app does not diagnose; this message comes from safety rules that are still waiting for review by a clinician.")
    return head + "\n\n" + "\n\n".join(f"- {x}" for x in lines) + "\n\n" + tail


def alert_block(res):
    """Text appended to the AI system prompt so the nurse knows what the floor found."""
    if not res.hits:
        return ""
    lines = [f"- [{h.level}] {h.id}: {h.en}" for h in res.hits]
    return ("SAFETY RULES (code-level, from what the user wrote). Never answer below these levels and never reassure "
            "against them:\n" + "\n".join(lines))


def fallback_reply(res=None, lang="en"):
    """Shown when the AI call fails: URGENT by default, the rule floor when it is higher. Never an all-clear."""
    el = lang == "el"
    if res and res.hits and res.level == "EMERGENCY":
        return act_now_reply(res, lang) if act_now_hits(res) else (
            ("🚨 **Κάλεσε τώρα 166 ή 112.**\n\n" if el else "🚨 **Call 166 or 112 now.**\n\n") +
            "\n\n".join(f"- {h.el if el else h.en}" for h in res.hits))
    base = ("⚠️ Δεν μπόρεσα να ολοκληρώσω την αξιολόγηση αυτή τη στιγμή. Θεώρησέ το **επείγον**: επικοινώνησε με γιατρό σήμερα. "
            "Αν έχεις πόνο στο στήθος, δυσκολία στην αναπνοή, σημεία εγκεφαλικού, έντονη αιμορραγία ή λιποθυμία, κάλεσε **166 ή 112** τώρα."
            if el else
            "⚠️ I could not finish the assessment right now. Please treat this as **urgent** and contact a doctor today. "
            "If you have chest pain, trouble breathing, signs of a stroke, severe bleeding or you faint, call **166 or 112** now.")
    if res and res.hits:
        base += "\n\n" + "\n".join(f"- {h.el if el else h.en}" for h in res.hits)
    return base


def is_ai_error(reply):
    """claude() returns '⚠️ ...' strings on failure (missing key, timeout, HTTP error)."""
    return bool(reply) and reply.lstrip().startswith("⚠️") and ("Claude" in reply[:40] or "Request timed out" in reply[:40])
