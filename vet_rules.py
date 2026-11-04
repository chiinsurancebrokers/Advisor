"""Deterministic veterinary safety rules for PetsAIHealth.

These rules are a SAFETY FLOOR under the nurse chat, not a replacement for it:
they can only RAISE the urgency of a final triage verdict, never lower it.
The LLM still reads the owner's free text and decides; this module catches
well-known red flags (toxins, GDV, urinary blockage, egg binding ...) with
the same answer every time and a traceable rule id + source.

Matching is done on accent-stripped, lower-cased text (final sigma folded to
sigma), so Greek and English wording both work. Patterns below are therefore
written WITHOUT accents and with plain sigma.

Sources: pages of the MSD (Merck) Veterinary Manual, paraphrased, never quoted.
Rules with basis="policy" come from the app's existing nurse prompt and have
no MSD page attached yet: they need review by a veterinarian.

Levels use the app's own tags: EMERGENCY > URGENT > SELF_CARE.
"""
import re
import unicodedata
from dataclasses import dataclass, field

RANK = {"SELF_CARE": 0, "URGENT": 1, "EMERGENCY": 2}

MSD = "https://www.msdvetmanual.com/"
SRC = {
    "gdv": MSD + "digestive-system/diseases-of-the-stomach-and-intestines-in-small-animals/gastric-dilation-and-volvulus-in-small-animals",
    "chocolate": MSD + "toxicology/food-hazards/chocolate-toxicosis-in-animals",
    "xylitol": "https://www.merckvetmanual.com/toxicology/food-hazards/xylitol-toxicosis-in-dogs",
    "grape": MSD + "toxicology/food-hazards/grape-raisin-and-tamarind-vitis-spp-tamarindus-spp-toxicosis-in-dogs",
    "glycol": "https://www.merckvetmanual.com/toxicology/ethylene-glycol-toxicosis/ethylene-glycol-toxicosis-in-animals",
    "lily": MSD + "toxicology/poisonous-plants/houseplants-and-ornamentals-toxic-to-animals",
    "urethral": "https://www.merckvetmanual.com/urinary-system/urolithiasis-in-small-animals/urethral-obstruction-in-small-animals",
    "analgesics": MSD + "toxicology/toxicoses-from-human-analgesics/toxicoses-from-human-analgesics-in-animals",
    "rodenticide": MSD + "toxicology/rodenticide-poisoning/anticoagulant-rodenticide-poisoning-in-animals",
    "rabbit_gi": MSD + "all-other-pets/rabbits/digestive-disorders-of-rabbits",
    "bird_repro": MSD + "exotic-and-laboratory-animals/pet-birds/reproductive-diseases-of-pet-birds",
    "bird_owner": "https://www.merckvetmanual.com/bird-owners/disorders-and-diseases-of-birds/reproductive-disorders-of-pet-birds",
    "reptile": "https://www.merckvetmanual.com/all-other-pets/reptiles/disorders-and-diseases-of-reptiles",
    # pet-owner section pages
    "snake": MSD + "special-pet-topics/poisoning/snakebite",
    "metaldehyde": MSD + "special-pet-topics/poisoning/metaldehyde-poisoning",
    "insecticide": MSD + "special-pet-topics/poisoning/insecticide-poisoning",
    "food": MSD + "special-pet-topics/poisoning/food-hazards",
    "household": MSD + "special-pet-topics/poisoning/household-hazards",
    "otc": MSD + "special-pet-topics/poisoning/poisoning-from-human-over-the-counter-drugs",
    "rx": MSD + "special-pet-topics/poisoning/poisonings-from-human-prescription-drugs",
    "plants": MSD + "special-pet-topics/poisoning/plants-poisonous-to-animals",
    "salt": MSD + "special-pet-topics/poisoning/salt-poisoning",
    "emergency": MSD + "special-pet-topics/emergencies/what-to-do-in-a-dog-or-cat-emergency",
    "dog_gi": MSD + "dog-owners/digestive-disorders-of-dogs/disorders-of-the-stomach-and-intestines-in-dogs",
    "gp_common": MSD + "all-other-pets/guinea-pigs/common-health-problems-of-guinea-pigs",
    "gp_nutrition": MSD + "all-other-pets/guinea-pigs/nutritional-problems-of-guinea-pigs",
    "ferret_emerg": MSD + "all-other-pets/ferrets/emergencies-of-ferrets",
    "hamster": MSD + "all-other-pets/hamsters/disorders-and-diseases-of-hamsters",
    "cases": MSD + "pages-with-widgets/case-studies",
}


def norm(text):
    """Lower-case, strip diacritics, fold final sigma."""
    if not text:
        return ""
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("ς", "σ")


@dataclass
class Rule:
    id: str
    species: tuple          # () = every species
    level: str
    all_of: tuple           # every regex must match
    en: str
    el: str
    source: str = ""
    basis: str = "msd"      # "msd" or "policy"
    exclude: str = ""       # if this matches, the rule does not fire
    immediate: bool = False  # clear immediate danger: nurse may skip the question cycle
    neg_guard: bool = False  # exposure rule: ignore clauses where the owner says it did NOT happen ("didn't eat the grapes")


@dataclass
class Hit:
    id: str
    level: str
    en: str
    el: str
    source: str
    basis: str
    immediate: bool = False


@dataclass
class Result:
    level: str = ""
    hits: list = field(default_factory=list)


def _r(*patterns):
    return tuple(re.compile(p, re.S) for p in patterns)


# ---- shared fragments ------------------------------------------------------
_SEIZ = (r"seizur|convuls|paddl|thrash|epilep|\bfitting\b|"
         r"σπασμ|επιληπ|σπαρταρ")
_LONG = (r"(lasted|lasting|for|going|διαρκεσ\w*|εδω και|ηδη) .{0,10}(\b([5-9]|[1-9]\d)|five|six|seven|eight|nine|ten|fifteen|twenty|πεντε|εξι|εφτα|οκτω|εννια|δεκα|δεκαπεντε|εικοσι) ?(min|λεπτ)|half an hour|several minutes|more than (five|5)|still (seizing|having|fitting)|"
         r"won'?t stop|not stopping|cluster|repeat|again and again|one after another|second seizure|third seizure|"
         r"πολλα λεπτα|δεν σταματ|ξανα και ξανα|επαναλαμβαν|πολλαπλ|διαρκουν")
_GUMS_BAD = (r"(pale|white|blue|bluish|grey|gray|purple|cyan|muddy|brown)\w*.{0,15}gums?|"
             r"gums?.{0,25}(pale|white|blue|bluish|grey|gray|purple|cyan|muddy)|"
             r"(χλωμ|ασπρ|μπλε|γαλαζ|γκρι|μωβ)\w*.{0,15}ουλα|ουλα.{0,25}(χλωμ|ασπρ|μπλε|γαλαζ|γκρι|μωβ)")
_INGEST = (r"\b(ate|eaten|eats|eating|swallowed?|ingest\w*|chew\w*|nibbl\w*|licked|lapped|drank|drunk|stole|stolen|took|taken|gave|given)\b|got into|got hold of|"
           r"εφαγε|φαει|καταπι|μασησε|μασουλ|τσιμπ|γλειψ|ηπιε|εκλεψε|πηρε|εδωσα")

RULES = [
    # ---------------------------------------------------------------- dogs
    Rule("DOG-GDV", ("dog",), "EMERGENCY",
         _r(r"(swollen|bloat|distend|balloon|drum|hard|tight)\w*.{0,40}(belly|abdomen|stomach|tummy)|"
            r"(belly|abdomen|stomach|tummy).{0,40}(swollen|bloat|distend|balloon|drum|hard|tight)|"
            r"φουσκ|πρησ|σκληρ\w* κοιλι|κοιλι\w*.{0,30}(σκληρ|τεντωμ|φουσκ|πρησ)|στομαχ\w*.{0,30}(φουσκ|πρησ)",
            r"retch|unproductive|nothing (comes|came) (up|out)|trying to (throw up|vomit)|dry heav|drool|restless|pacing|"
            r"ξερομ|ξερατ|ξερασ|θελει να (κανει εμετο|ξερασει)|δεν βγαινει τιποτα|σιελορ|σαλια|ανησυχ|βηματιζ"),
         "Swollen belly with retching or restlessness can be gastric torsion (GDV): it needs treatment within hours.",
         "Φουσκωμένη κοιλιά με ξερατά ή ανησυχία μπορεί να είναι στροφή στομάχου: χρειάζεται θεραπεία μέσα σε ώρες.",
         SRC["gdv"], "msd", immediate=True),
    Rule("DOG-XYLITOL", ("dog",), "EMERGENCY",
         _r(r"xylitol|ξυλιτολ|(sugar|ζαχαρ\w*)[- ]?free (gum|candy|mint)|χωρις ζαχαρη.{0,25}(τσιχλ|καραμελ)|"
            r"τσιχλ\w* χωρις ζαχαρη"),
         "Xylitol can drop a dog's blood sugar within 30 minutes and later damage the liver.",
         "Η ξυλιτόλη μπορεί να ρίξει το σάκχαρο του σκύλου μέσα σε 30 λεπτά και αργότερα να βλάψει το συκώτι.",
         SRC["xylitol"], "msd", immediate=True),
    Rule("DOG-GRAPES", ("dog",), "EMERGENCY",
         _r(r"grape|raisin|sultana|currant|σταφυλ|σταφιδ|κορινθιακ"),
         "No safe amount of grapes or raisins is known for dogs: kidney failure can follow, and early treatment matters.",
         "Δεν υπάρχει γνωστή ασφαλής ποσότητα σταφυλιού ή σταφίδας για σκύλους: μπορεί να ακολουθήσει νεφρική ανεπάρκεια και η έγκαιρη θεραπεία μετράει.",
         SRC["grape"], "msd", exclude=r"grape.{0,3}flavo|flavo.{0,10}grape|grape.?seed|γευση σταφυλ|σταφυλ\w* γευση",
         immediate=True),
    # ---------------------------------------------------------------- cats
    Rule("CAT-LILY", ("cat",), "EMERGENCY",
         _r(r"\blil(y|ies)\b|lilium|daylil|hemerocallis|easter lil|tiger lil|κρινο|κρινου|κρινοι|ημεροκαλλ"),
         "True lilies and daylilies can cause acute kidney failure in cats, even from a small amount; early treatment gives the best outcome.",
         "Οι αληθινοί κρίνοι και οι ημεροκαλλίδες μπορούν να προκαλέσουν οξεία νεφρική ανεπάρκεια στις γάτες, ακόμη και από μικρή ποσότητα· η έγκαιρη θεραπεία δίνει το καλύτερο αποτέλεσμα.",
         SRC["lily"], "msd", exclude=r"peace lil|calla|spathiph|σπαθιφυλλ|arum|αρον", immediate=True),
    Rule("CAT-FLOWERS", ("cat",), "URGENT",
         _r(r"bouquet|flowers?|petals?|ανθοδεσμ|λουλουδ|πεταλ",
            r"chew|nibbl|ate|eat|licked|τσιμπ|εφαγε|μασ|γλειψ"),
         "A cat that chewed flowers may have eaten a lily: find out which flowers they were and call a vet today.",
         "Γάτα που μάσησε λουλούδια μπορεί να έφαγε κρίνο: βρες ποια λουλούδια ήταν και κάλεσε κτηνίατρο σήμερα.",
         SRC["lily"], "msd", exclude=r"peace lil|calla|spathiph|σπαθιφυλλ"),
    Rule("CAT-URETHRAL", ("cat",), "EMERGENCY",
         _r(r"strain|squat|trying to (pee|urinate|wee)|(going|goes|went|visiting|visits).{0,25}(litter|box|tray)|keeps going to the (litter|box|tray)|"
            r"litter|προσπαθει να ουρησει|(πηγαινει|παει|ξαναπηγαινει).{0,25}(αμμο|λεκανη)|λεκανη|αμμο",
            r"nothing (comes|came|is coming|coming|happen\w*|out|to show)|passing nothing|producing nothing|no urine|not producing|only (a |the )?(few |tiny )?(drop|dribble|bit)|just (a )?(drop|dribble)|"
            r"drops? .{0,12}(two|few)|can'?t (pee|urinate|wee)|unable to (pee|urinate)|"
            r"δεν (βγαινει|βγαζει|ουρει|καταφερνει)|σταγον|καθολου ουρ|τιποτα (δεν )?(βγαινει|βγαζει)|(βγαινει|βγαζει|βγηκε) τιποτα|ελαχιστο ουρο"),
         "Straining to urinate with little or no urine can be a urethral blockage, which can be fatal within about 2 to 3 days and is an emergency.",
         "Προσπάθεια για ούρηση με λίγα ή καθόλου ούρα μπορεί να είναι απόφραξη της ουρήθρας, που μπορεί να είναι θανατηφόρα σε περίπου 2-3 μέρες και είναι επείγον.",
         SRC["urethral"], "msd", immediate=True),
    Rule("CAT-PARACETAMOL", ("cat",), "EMERGENCY",
         _r(r"paracetamol|acetaminophen|tylenol|panadol|depon|ντεπον|παρακεταμολ"),
         "Cats are very sensitive to paracetamol: toxic effects can start at low doses and affect the blood within hours.",
         "Οι γάτες είναι πολύ ευαίσθητες στην παρακεταμόλη: τοξικά αποτελέσματα μπορεί να ξεκινήσουν με μικρές δόσεις και να επηρεάσουν το αίμα μέσα σε ώρες.",
         SRC["analgesics"], "msd", immediate=True),
    # ------------------------------------------------------ dogs and cats
    Rule("DC-NSAID", ("dog", "cat"), "URGENT",
         _r(r"ibuprofen|nurofen|advil|naproxen|aleve|brufen|ιβουπροφ|νουροφεν|ναπροξ|μπρουφεν",
            _INGEST),
         "Human anti-inflammatory pills (ibuprofen, naproxen) can cause stomach ulcers and kidney damage in dogs and cats: call a vet today.",
         "Τα ανθρώπινα αντιφλεγμονώδη (ιβουπροφαίνη, ναπροξένη) μπορεί να προκαλέσουν έλκη στομάχου και νεφρική βλάβη σε σκύλους και γάτες: κάλεσε κτηνίατρο σήμερα.",
         SRC["analgesics"], "msd"),
    Rule("DC-ANTIFREEZE", ("dog", "cat"), "EMERGENCY",
         _r(r"antifreeze|anti-freeze|ethylene glycol|coolant|αντιψυκτικ|αιθυλενογλυκολ|ψυκτικ\w* υγρ"),
         "Antifreeze (ethylene glycol) is deadly, and treatment only works well if it starts before the kidneys fail.",
         "Το αντιψυκτικό (αιθυλενογλυκόλη) είναι θανατηφόρο και η θεραπεία δουλεύει καλά μόνο αν ξεκινήσει πριν αποτύχουν τα νεφρά.",
         SRC["glycol"], "msd", immediate=True),
    Rule("DC-RODENTICIDE", ("dog", "cat"), "URGENT",
         _r(r"rat poison|rodenticide|mouse poison|warfarin|brodifacoum|bromadiolone|ποντικοφαρμακ|φαρμακο για ποντικ|δολωμα|ραδεντ|βρωμαδιολον",
            _INGEST),
         "Rat poison stops blood clotting and bleeding shows up days later; a vet can act before that if the pet is seen soon.",
         "Το ποντικοφάρμακο σταματά την πήξη του αίματος και η αιμορραγία φαίνεται μέρες αργότερα· ο κτηνίατρος μπορεί να δράσει πριν, αν δει το ζώο νωρίς.",
         SRC["rodenticide"], "msd"),
    Rule("DC-RODENTICIDE-BLEED", ("dog", "cat"), "EMERGENCY",
         _r(r"rat poison|rodenticide|mouse poison|warfarin|brodifacoum|bromadiolone|ποντικοφαρμακ|φαρμακο για ποντικ|δολωμα|ραδεντ|βρωμαδιολον",
            r"blood|bleed|bruis|nosebleed|black stool|αιμα|αιμορραγ|μελανιασ|μαυρα κοπρανα"),
         "Possible rat poison plus any bleeding is an emergency.",
         "Πιθανό ποντικοφάρμακο μαζί με οποιαδήποτε αιμορραγία είναι επείγον.",
         SRC["rodenticide"], "msd", immediate=True),
    # -------------------------------------------------------------- rabbits
    Rule("RAB-STASIS", ("rabbit",), "URGENT",
         _r(r"not eating|stopped eating|won'?t eat|no appetite|hasn'?t eaten|refus\w* (food|hay|pellet)|no (poop|droppings|faeces|feces)|"
            r"fewer (poop|droppings)|not pooping|(hasn.?t|haven.?t|not|stopped) (pooped|pooping|passed (any )?(poop|droppings|stool))|"
            r"δεν τρωει|σταματησε να τρωει|δεν (κανει|βγαζει) κοπρανα|λιγοτερα κοπρανα|χωρις κοπρανα|δεν τρωει τιποτα"),
         "A rabbit that stops eating or pooping needs a vet promptly: gut stasis can become life-threatening, so do not wait and see.",
         "Κουνέλι που σταματά να τρώει ή να κάνει κόπρανα χρειάζεται κτηνίατρο άμεσα: η στάση του εντέρου μπορεί να γίνει απειλητική για τη ζωή, μην περιμένεις.",
         SRC["rabbit_gi"], "msd"),
    Rule("RAB-STASIS-SEVERE", ("rabbit",), "EMERGENCY",
         _r(r"not eating|stopped eating|won'?t eat|no appetite|hasn'?t eaten|no (poop|droppings|faeces|feces)|not pooping|(hasn.?t|haven.?t|not|stopped) (pooped|pooping|passed (any )?(poop|droppings|stool))|δεν τρωει|σταματησε να τρωει|χωρις κοπρανα|δεν (κανει|βγαζει) κοπρανα",
            r"bloat|swollen|lethar|cold|weak|grind|hunched|collaps|φουσκ|πρησ|ληθαργ|κρυο|αδυναμ|τριξιμο δοντι|κουλουριασμ"),
         "A rabbit that stopped eating and is bloated, cold, weak or in pain needs a vet right now.",
         "Κουνέλι που σταμάτησε να τρώει και είναι φουσκωμένο, κρύο, αδύναμο ή πονάει χρειάζεται κτηνίατρο τώρα.",
         SRC["rabbit_gi"], "msd", immediate=True),
    Rule("RAB-DIARRHEA", ("rabbit",), "URGENT",
         _r(r"diarrh|watery (stool|poop)|διαρροια|νερουλ\w* κοπρανα"),
         "Diarrhoea in a rabbit, at any age, is serious and needs a vet.",
         "Η διάρροια σε κουνέλι, σε οποιαδήποτε ηλικία, είναι σοβαρή και χρειάζεται κτηνίατρο.",
         SRC["rabbit_gi"], "msd"),
    Rule("RAB-DIARRHEA-YOUNG", ("rabbit",), "EMERGENCY",
         _r(r"diarrh|watery (stool|poop)|διαρροια|νερουλ\w* κοπρανα",
            r"baby|young|weeks? old|kit\b|just weaned|weaned|μωρο|μικρο|εβδομαδ|απογαλακτ"),
         "Sudden severe diarrhoea in a young rabbit can kill quickly: go to a vet now.",
         "Ξαφνική έντονη διάρροια σε νεαρό κουνέλι μπορεί να σκοτώσει γρήγορα: πήγαινε σε κτηνίατρο τώρα.",
         SRC["rabbit_gi"], "msd", immediate=True),
    # ---------------------------------------------------------------- birds
    Rule("BIRD-EGGBOUND", ("bird",), "EMERGENCY",
         _r(r"\begg|laying|lay an egg|gravid|αυγ|ωοτοκ",
            r"strain|floor|fluff|tail[- ]?bob|swollen|distend|weak|lethar|sitting|φουσκ|πρηξ|δυσκολ|κουναει την ουρα|πατωμα|ληθαργ|αδυναμ|κατω"),
         "A bird with signs of being egg-bound (on the cage floor, straining, swollen belly, weak) is an emergency.",
         "Πουλί με σημεία εγκλωβισμένου αυγού (στο πάτωμα του κλουβιού, πίεση, φουσκωμένη κοιλιά, αδυναμία) είναι επείγον.",
         SRC["bird_repro"], "msd", immediate=True),
    Rule("BIRD-TAILBOB", ("bird",), "EMERGENCY",
         _r(r"tail[- ]?bob|tail.{0,12}bobbing|bobbing.{0,12}tail|κουναει την ουρα|κουνιεται η ουρα|ουρα.{0,15}πανω κατω|labou?red breath|δυσκολια.{0,10}αναπν"),
         "Tail bobbing or laboured breathing in a bird means it is struggling to breathe: go to an avian vet now.",
         "Το κούνημα της ουράς ή η δύσκολη αναπνοή σε πουλί σημαίνει ότι δυσκολεύεται να αναπνεύσει: πήγαινε σε κτηνίατρο πουλιών τώρα.",
         SRC["bird_owner"], "msd", immediate=True),
    Rule("BIRD-REPTILE-PROLAPSE", ("bird", "reptile"), "EMERGENCY",
         _r(r"prolaps|tissue (protrud|sticking|hanging)|(protrud|sticking out|hanging out).{0,20}(vent|cloaca)|προπτωσ|προβαλλ\w*.{0,20}(αμαρα|κλοακ)"),
         "Tissue or organs protruding from the vent need immediate veterinary care.",
         "Ιστός ή όργανα που προβάλλουν από την κλοάκα χρειάζονται άμεση κτηνιατρική φροντίδα.",
         SRC["bird_owner"] + " | " + SRC["reptile"], "msd", immediate=True),
    # ------------------------------------------------------------- reptiles
    Rule("REP-RESP", ("reptile",), "URGENT",
         _r(r"open[- ]?mouth|mouth open|nasal discharge|runny nose|bubbl\w* (at|from) (the )?(nose|mouth)|wheez|labou?red breath|"
            r"ανοιχτο στομα|εκκριση απο τη μυτη|φταρνισ|συριγμ"),
         "Open-mouth breathing or nasal discharge in a reptile can mean a respiratory infection: see a reptile vet promptly.",
         "Αναπνοή με ανοιχτό στόμα ή ρινική έκκριση σε ερπετό μπορεί να σημαίνει λοίμωξη αναπνευστικού: δες κτηνίατρο ερπετών άμεσα.",
         SRC["reptile"], "msd"),
    Rule("REP-EGGS", ("reptile",), "URGENT",
         _r(r"\begg|gravid|retained|laying|αυγ|ωοτοκ",
            r"strain|not laying|can'?t lay|stuck|swollen|lethar|weak|δυσκολ|δεν γεννα|κολλησ|φουσκ|ληθαργ|αδυναμ"),
         "A reptile that cannot lay its eggs may be egg-bound (dystocia): this needs a vet, and surgery is often necessary.",
         "Ερπετό που δεν μπορεί να γεννήσει τα αυγά του μπορεί να έχει δυστοκία: χρειάζεται κτηνίατρο και συχνά χειρουργείο.",
         SRC["reptile"], "msd"),
    Rule("REP-NEURO", ("reptile",), "URGENT",
         _r(r"star ?gazing|twisted neck|head tilt|loss of (movement|control)|can'?t (move|walk)|κεφαλι.{0,12}γυρισμεν|δεν μπορει να περπατ|δεν κουναει"),
         "A twisted neck, stargazing or loss of movement in a reptile can mean a neurological problem or metabolic bone disease: see a vet promptly.",
         "Στριμμένος λαιμός ή απώλεια κίνησης σε ερπετό μπορεί να σημαίνει νευρολογικό πρόβλημα ή μεταβολική νόσο των οστών: δες κτηνίατρο άμεσα.",
         SRC["reptile"], "msd"),
    # ------------------------------------------------------- every species
    Rule("ALL-SEIZURE-LONG", (), "EMERGENCY",
         _r(_SEIZ, _LONG),
         "A seizure lasting about 5 minutes or more, or repeating, is an emergency.",
         "Σπασμός που διαρκεί περίπου 5 λεπτά ή περισσότερο, ή που επαναλαμβάνεται, είναι επείγον.",
         "", "policy", immediate=True),
    Rule("ALL-SEIZURE", (), "URGENT",
         _r(_SEIZ),
         "Any seizure needs a vet check today.",
         "Κάθε σπασμός χρειάζεται εξέταση από κτηνίατρο σήμερα.",
         "", "policy"),
    Rule("ALL-GUMS", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(_GUMS_BAD),
         "Pale, white, blue or grey gums can mean shock, blood loss or lack of oxygen: go to a vet now.",
         "Χλωμά, λευκά, μπλε ή γκρίζα ούλα μπορεί να σημαίνουν σοκ, απώλεια αίματος ή έλλειψη οξυγόνου: πήγαινε σε κτηνίατρο τώρα.",
         "", "policy", immediate=True),
    Rule("ALL-BREATH", ("dog", "cat", "rabbit", "bird", "other"), "EMERGENCY",
         _r(r"gasping|struggling to breathe|can'?t breathe|cannot breathe|not breathing|blue tongue|"
            r"δεν μπορει να αναπνε|δεν αναπνε|ασφυξ|δυσκολια στην αναπνο|αγωνια.{0,15}αναπν"),
         "Struggling to breathe is an emergency.",
         "Η δυσκολία στην αναπνοή είναι επείγον.",
         "", "policy", immediate=True),
    Rule("CAT-OPENMOUTH", ("cat",), "EMERGENCY",
         _r(r"open[- ]?mouth.{0,15}breath|breath\w*.{0,15}open[- ]?mouth|mouth open|panting|ανοιχτο στομα.{0,15}αναπν|αναπν\w*.{0,15}ανοιχτο στομα|λαχανιαζ"),
         "Open-mouth breathing or panting in a cat is an emergency.",
         "Αναπνοή με ανοιχτό στόμα ή λαχάνιασμα σε γάτα είναι επείγον.",
         "", "policy", immediate=True),
    Rule("ALL-COLLAPSE", (), "EMERGENCY",
         _r(r"collaps|unconscious|unresponsive|not responding|won'?t wake|fainted|passed out|can'?t stand|cannot stand|fell over|"
            r"καταρρευσ|καταρρε|κατερρε|λιποθυμ|αναισθητ|δεν ανταποκρινεται|δεν μπορει να σταθει|επεσε κατω"),
         "Collapse, fainting or being unable to stand is an emergency.",
         "Κατάρρευση, λιποθυμία ή αδυναμία να σταθεί είναι επείγον.",
         "", "policy", immediate=True),
    Rule("ALL-TRAUMA-BLEED", (), "EMERGENCY",
         _r(r"hit by (a )?(car|vehicle)|run over|road accident|car accident|fell (from|off).{0,20}(balcony|window|roof|height|tree)|fall from|"
            r"bleeding (heavily|a lot|profusely)|won'?t stop bleeding|bleeding (that )?won'?t stop|"
            r"αυτοκινητ.{0,15}(χτυπ|παρεσυρ)|χτυπηθηκε (απο|σε) (αυτοκιν|αμαξι|οχημα|μηχαν)|(τον|την|το) χτυπησε (αυτοκιν|αμαξι|οχημα|μηχαν)|παρασυρθηκε|(τον|την|το) παρεσυρε|τροχαι|επεσε απο.{0,15}(μπαλκον|παραθυρ|ταρατσ|υψος)|δεν σταματαει το αιμα|αιμορραγ\w*.{0,20}δεν σταματ"),
         "A road accident, a fall from height or bleeding that will not stop is an emergency, even if the pet looks fine.",
         "Τροχαίο, πτώση από ύψος ή αιμορραγία που δεν σταματά είναι επείγον, ακόμη κι αν το ζώο φαίνεται καλά.",
         "", "policy", immediate=True),
    Rule("ALL-HEATSTROKE", (), "EMERGENCY",
         _r(r"heat ?stroke|overheat|left in (the |a )?(hot )?car|hot car|θερμοπληξ|υπερθερμ|κλειδωμεν\w* (στο )?αυτοκινητ"),
         "Heatstroke or overheating is an emergency.",
         "Η θερμοπληξία ή η υπερθέρμανση είναι επείγον.",
         "", "policy", immediate=True),
    Rule("ALL-EYE-PROPTOSIS", (), "EMERGENCY",
         _r(r"eye.{0,15}(popped|out of (the )?socket|bulging out|proptos)|προεκβολ\w* οφθαλμ|ματι.{0,15}εβγηκε|ματι.{0,20}εξω απο"),
         "An eye out of its socket is an emergency.",
         "Μάτι που έχει βγει από την κόγχη είναι επείγον.",
         "", "policy", immediate=True),
    Rule("MAM-DYSTOCIA", ("dog", "cat", "rabbit"), "EMERGENCY",
         _r(r"(labou?r|giving birth|whelping|queening).{0,40}(hours?|no (puppy|kitten)|stuck|green discharge)|straining.{0,40}(no (puppy|kitten)|stuck|green discharge|in labou?r|pregnant)|"
            r"τοκετ.{0,30}(ωρες|κολλησε|πρασιν)|γεννα\w*.{0,30}(ωρες|κολλησε)"),
         "Prolonged or stalled labour is an emergency.",
         "Παρατεταμένος ή σταματημένος τοκετός είναι επείγον.",
         "", "policy", immediate=True),
    # ================= added from the MSD pet-owner pages (food, household, drugs, plants, bites, emergencies)
    Rule("DC-SNAKEBITE", ("dog", "cat"), "EMERGENCY",
         _r(r"snake|viper|adder|\bφιδι|οχι[αά]|οχια", r"\bbit(e|es|ten)?\b|bitten|fang|strike|struck|δαγκ|τσιμπ"),
         "A snakebite is an emergency: keep the pet calm and still, go to a vet immediately, and do not cut, suck, ice or tie the wound.",
         "Δάγκωμα φιδιού είναι επείγον: κράτα το ζώο ήρεμο και ακίνητο, πήγαινε αμέσως σε κτηνίατρο και μην κόψεις, μην ρουφήξεις, μην βάλεις πάγο και μην δέσεις την πληγή.",
         SRC["snake"], "msd", immediate=True),
    Rule("DC-METALDEHYDE", ("dog", "cat"), "EMERGENCY",
         _r(r"metaldehyde|μεταλδεϋδ|μεταλδευδ|(slug|snail|σαλιγκαρ|γυμνοσαλιγκ)\w*.{0,25}(bait|pellet|poison|killer|δολωμ|φαρμακ|χαπακ|κοκκ)|(δολωμ|φαρμακ|bait|pellet)\w*.{0,20}(σαλιγκαρ|γυμνοσαλιγκ|slug|snail)", _INGEST),
         "Slug and snail bait (metaldehyde) causes tremors, fever and seizures and needs intensive treatment: go to a vet now.",
         "Το δόλωμα για σαλιγκάρια και γυμνοσάλιαγκες (μεταλδεΰδη) προκαλεί τρέμουλο, πυρετό και σπασμούς και θέλει εντατική θεραπεία: πήγαινε σε κτηνίατρο τώρα.",
         SRC["metaldehyde"], "msd", immediate=True),
    Rule("DC-POISON-BAIT", ("dog", "cat"), "EMERGENCY",
         _r(r"poison(ed)? bait|strychnine|aldicarb|methomyl|carbamate|organophosph|στρυχνιν|αλντικαρμπ|μεθομυλ|οργανοφωσφ|"
            r"δηλητηριασμεν\w*.{0,15}(δολωμ|φαγητ|κρεας|λουκανικ|κεφτε)"),
         "Suspected poisoned bait or an insecticide (carbamate, organophosphate, strychnine) can cause drooling, tremors, breathing trouble and seizures within hours: go to a vet now.",
         "Πιθανό δηλητηριασμένο δόλωμα ή εντομοκτόνο (καρβαμιδικό, οργανοφωσφορικό, στρυχνίνη) μπορεί να προκαλέσει σάλια, τρέμουλο, δύσπνοια και σπασμούς μέσα σε ώρες: πήγαινε σε κτηνίατρο τώρα.",
         SRC["insecticide"], "msd", immediate=True,
         exclude=r"\b(no|not any|without)\b .{0,12}poison|δεν υπηρχε.{0,15}δηλητηρ|χωρις δηλητηρ"),
    Rule("CAT-DOGFLEA", ("cat",), "EMERGENCY",
         _r(r"permethrin|περμεθριν|dog'?s? (flea|tick|spot|pipette|drops|treatment|collar|product)|(for|made for) dogs|σταγονες.{0,12}σκυλ|αντιπαρασιτικ\w*.{0,12}σκυλ|για σκυλους|για σκυλο",
            r"appl|put|used|gave|on (my|the|her|him|our) cat|εβαλα|εφαρμοσα|χρησιμοποιησα|στη γατα|στον γατο|στη γατουλα"),
         "Dog flea products (permethrin) can cause tremors and seizures in cats: wash the product off now with mild dish soap and lukewarm water, keep the cat warm, and go to a vet immediately.",
         "Τα αντιπαρασιτικά για σκύλους (περμεθρίνη) μπορεί να προκαλέσουν τρέμουλο και σπασμούς στις γάτες: ξέπλυνε αμέσως το προϊόν με ήπιο απορρυπαντικό πιάτων και χλιαρό νερό, κράτα τη γάτα ζεστή και πήγαινε αμέσως σε κτηνίατρο.",
         SRC["insecticide"], "policy", immediate=True),
    Rule("DC-BATTERY", ("dog", "cat"), "EMERGENCY",
         _r(r"batter(y|ies)|button cell|coin cell|μπαταρι", _INGEST),
         "A swallowed battery (especially a flat button or lithium cell) can burn the throat and stomach within hours: go to a vet now and do not make the pet vomit.",
         "Μια μπαταρία που καταπίθηκε (ιδίως κουμπί ή λιθίου) μπορεί να κάψει τον οισοφάγο και το στομάχι μέσα σε ώρες: πήγαινε σε κτηνίατρο τώρα και μην προκαλέσεις εμετό.",
         SRC["household"], "msd", immediate=True),
    Rule("DC-CORROSIVE", ("dog", "cat"), "URGENT",
         _r(r"drain (cleaner|opener|unblocker)|bleach|dishwasher (tablet|detergent|pod|powder)|toilet (cleaner|bowl cleaner)|oven cleaner|caustic|corrosive|limescale|"
            r"χλωριν|αλευκαντικ|μπλιτσ|καθαριστικ\w*.{0,12}(αποχετευσ|τουαλετ|φουρν|πλυντηρι)|αποφρακτικ|καυστικ", _INGEST),
         "Swallowed drain, toilet or dishwasher cleaner and bleach can burn the mouth, throat and stomach, sometimes hours later: call a vet now, give only water or milk, and do not induce vomiting.",
         "Καθαριστικά αποχέτευσης, τουαλέτας ή πλυντηρίου πιάτων και χλωρίνη που καταπίνονται μπορεί να κάψουν στόμα, λαιμό και στομάχι, ακόμη και ώρες αργότερα: κάλεσε κτηνίατρο τώρα, δώσε μόνο νερό ή γάλα και μην προκαλέσεις εμετό.",
         SRC["household"], "msd"),
    Rule("DC-ALCOHOL", ("dog", "cat"), "URGENT",
         _r(r"alcohol|\bbeer|\bwine|vodka|whisk|\brum\b|ouzo|tsipouro|raki\b|liqueur|antiseptic spray|αλκοολ|μπυρα|κρασι|βοτκα|ουισκι|ουζο|τσιπουρο|ρακη",
            r"drank|drunk|lapped|licked|\bate\b|swallow|ηπιε|ηπιαν|γλειψ|καταπι|ποτισ|εφαγε"),
         "Alcohol can cause vomiting, wobbliness, low body temperature and slow breathing within 30 to 60 minutes: call a vet now.",
         "Το αλκοόλ μπορεί να προκαλέσει εμετό, αστάθεια, χαμηλή θερμοκρασία και αργή αναπνοή μέσα σε 30 έως 60 λεπτά: κάλεσε κτηνίατρο τώρα.",
         SRC["household"], "msd"),
    Rule("DC-OTC-STIMULANT", ("dog", "cat"), "URGENT",
         _r(r"pseudoephedrine|ephedrine|ephedra|sudafed|decongestant|oxymetazoline|xylometazoline|naphazoline|tetrahydrozoline|nasal spray|eye drops for redness|guarana|"
            r"aspirin|ασπιριν|ψευδοεφεδριν|εφεδριν|ρινικο σπρει|αποσυμφορητικ", _INGEST),
         "Cold and flu medicines (decongestants), aspirin and stimulant supplements can cause a racing heart, tremors, seizures or bleeding in pets: call a vet or poison line now.",
         "Φάρμακα για το κρυολόγημα (αποσυμφορητικά), ασπιρίνη και διεγερτικά συμπληρώματα μπορεί να προκαλέσουν ταχυκαρδία, τρέμουλο, σπασμούς ή αιμορραγία: κάλεσε κτηνίατρο ή κέντρο δηλητηριάσεων τώρα.",
         SRC["otc"], "msd"),
    Rule("DC-RX-HUMAN", ("dog", "cat"), "URGENT",
         _r(r"antidepress|prozac|fluoxetine|sertraline|zoloft|citalopram|escitalopram|venlafaxine|amitriptyline|beta[- ]?blocker|propranolol|metoprolol|atenolol|bisoprolol|"
            r"amlodipine|diltiazem|verapamil|lisinopril|enalapril|ramipril|losartan|baclofen|cyclobenzaprine|zolpidem|alprazolam|xanax|diazepam|lorazepam|valium|"
            r"sleeping (pill|tablet|aid)|sedative|tranquili[sz]er|adhd|ritalin|adderall|amphetamine|methylphenidate|vitamin d|cholecalciferol|calcipotriene|calcipotriol|psoriasis|"
            r"iron (pill|tablet|supplement)|ferrous|blood pressure (pill|tablet|medic)|heart (pill|tablet|medic)|"
            r"αντικαταθλιπτ|ηρεμιστικ|υπνωτικ|βιταμινη d|σιδηρ|ψωριασ|χαπι\w*.{0,12}(πιεση|καρδια|υπνο|ηρεμ)|βηταλυτικ|ριταλιν", _INGEST),
         "Human prescription drugs (antidepressants, heart and blood-pressure pills, sleeping pills, ADHD drugs, vitamin D or iron) can cause heart rhythm problems, seizures or kidney damage, sometimes after a delay: call a vet or poison line now.",
         "Τα ανθρώπινα συνταγογραφούμενα φάρμακα (αντικαταθλιπτικά, χάπια για καρδιά και πίεση, υπνωτικά, φάρμακα ΔΕΠΥ, βιταμίνη D ή σίδηρος) μπορεί να προκαλέσουν αρρυθμίες, σπασμούς ή νεφρική βλάβη, ενίοτε με καθυστέρηση: κάλεσε κτηνίατρο ή κέντρο δηλητηριάσεων τώρα.",
         SRC["rx"], "msd"),
    Rule("DC-FLUOROURACIL", ("dog", "cat"), "EMERGENCY",
         _r(r"fluorouracil|5-?fu\b|efudex|carac\b|φθοριουρακιλ|φλουορουρακιλ", _INGEST),
         "5-fluorouracil skin cream can kill a dog even in small amounts, with vomiting then seizures within hours: go to a vet immediately.",
         "Η κρέμα 5-φθοριουρακίλης μπορεί να σκοτώσει σκύλο ακόμη και σε μικρή ποσότητα, με εμετό και μετά σπασμούς μέσα σε ώρες: πήγαινε αμέσως σε κτηνίατρο.",
         SRC["rx"], "msd", immediate=True),
    Rule("DC-PLANT-SEVERE", ("dog", "cat"), "EMERGENCY",
         _r(r"oleander|πικροδαφν|\byew\b|ταξοσ|lily of the valley|κρινακι.{0,12}κοιλαδ|castor (bean|oil plant)|\bricin|κικινο|azalea|rhododendron|αζαλεα|ροδοδενδρ|"
            r"mistletoe|ιξοσ|autumn crocus|colchicum|κολχικ", _INGEST),
         "Oleander, yew, lily of the valley, azalea, castor bean, mistletoe and autumn crocus are dangerous to the heart, gut or breathing and can kill quickly: go to a vet now.",
         "Η πικροδάφνη, ο ίταμος, το κρινάκι της κοιλάδας, η αζαλέα, η καστορέλαιο-φυτό (κικίνος), το γκι και το φθινοπωρινό κρόκο είναι επικίνδυνα για καρδιά, έντερο ή αναπνοή και μπορούν να σκοτώσουν γρήγορα: πήγαινε σε κτηνίατρο τώρα.",
         SRC["plants"], "msd", immediate=True),
    Rule("DC-PLANT-OTHER", ("dog", "cat"), "URGENT",
         _r(r"foxglove|διγιταλ|sago|cycad|coontie|κυκασ|dieffenbachia|dumb ?cane|philodendron|caladium|ντιφενμπαχ|φιλοδενδρ|καλαδι|"
            r"marijuana|cannabis|hashish|\bthc\b|κανναβ|μαριχουαν|χασισ|jerusalem cherry|cyclamen|κυκλαμιν", _INGEST),
         "This plant (or cannabis) can cause heart or liver problems, mouth and throat swelling, or poisoning signs within hours: call a vet today, sooner if breathing, wobbling or vomiting starts.",
         "Αυτό το φυτό (ή η κάνναβη) μπορεί να προκαλέσει καρδιακά ή ηπατικά προβλήματα, πρήξιμο σε στόμα και λαιμό ή σημεία δηλητηρίασης μέσα σε ώρες: κάλεσε κτηνίατρο σήμερα, νωρίτερα αν αρχίσει δύσπνοια, αστάθεια ή εμετός.",
         SRC["plants"], "msd"),
    Rule("DOG-BREADDOUGH", ("dog",), "URGENT",
         _r(r"\bdough\b|ζυμη|ζυμαρι|προζυμ", _INGEST),
         "Raw yeast dough swells in the stomach and makes alcohol: it can twist the stomach or cause staggering and seizures, so call a vet today and watch for a swollen belly or retching.",
         "Η ωμή ζύμη με μαγιά φουσκώνει στο στομάχι και παράγει αλκοόλ: μπορεί να στρίψει το στομάχι ή να προκαλέσει ζάλη και σπασμούς, οπότε κάλεσε κτηνίατρο σήμερα και πρόσεξε φουσκωμένη κοιλιά ή ξερομάρισμα.",
         SRC["food"], "msd"),
    Rule("BIRD-AVOCADO", ("bird",), "EMERGENCY",
         _r(r"avocado|guacamole|αβοκαντο", _INGEST),
         "Avocado can kill pet birds within 48 hours by damaging the heart: go to an avian vet now.",
         "Το αβοκάντο μπορεί να σκοτώσει πουλιά μέσα σε 48 ώρες βλάπτοντας την καρδιά: πήγαινε τώρα σε κτηνίατρο πουλιών.",
         SRC["food"], "msd", immediate=True),
    Rule("DC-SALT", ("dog", "cat"), "URGENT",
         _r(r"salt dough|play ?dough|rock salt|too much salt|(a lot|lots|bag|handful|pack) of salt|ζυμαρι.{0,12}αλατ|πολυ αλατι|αλατι.{0,12}(παλαμ|πακετο|σακουλ)", _INGEST),
         "A large amount of salt (salt dough, rock salt) can cause vomiting, tremors and seizures: call a vet now and give water only in small sips.",
         "Μεγάλη ποσότητα αλατιού (ζύμη αλατιού, χονδρό αλάτι) μπορεί να προκαλέσει εμετό, τρέμουλο και σπασμούς: κάλεσε κτηνίατρο τώρα και δώσε νερό μόνο σε μικρές γουλιές.",
         SRC["salt"], "msd"),
    Rule("DC-CHOKING", ("dog", "cat"), "EMERGENCY",
         _r(r"\bchok(e|es|ed|ing)\b|something stuck in (his|her|the|its) throat|stuck in (his|her|the|its) throat|πνιγεται|πνιγηκε|κολλησε.{0,15}λαιμο|σφηνωσε.{0,15}λαιμο"),
         "A choking pet can stop breathing: check the mouth without putting your fingers in, give firm thumps between the shoulder blades, and go to a vet at once.",
         "Ένα ζώο που πνίγεται μπορεί να σταματήσει να αναπνέει: κοίταξε το στόμα χωρίς να βάλεις τα δάχτυλα μέσα, δώσε δυνατά χτυπήματα ανάμεσα στις ωμοπλάτες και πήγαινε αμέσως σε κτηνίατρο.",
         SRC["emergency"], "msd", exclude=r"choke (chain|collar|leash)|chok\w* collar|κολαρο στραγγαλ", immediate=True),
    Rule("DC-BLOODY-GI", ("dog", "cat"), "URGENT",
         _r(r"(bloody|blood in|with blood|blood\b).{0,20}(diarrh|stool|poo|vomit|faeces|feces|vomit)|(diarrh|stool|poo|vomit|faeces|feces).{0,20}(blood|bloody|bloodstain)|coffee[- ]?ground|"
            r"αιμα.{0,20}(διαρρο|κοπρανα|εμετ)|(διαρροι|κοπρανα|εμετ\w*|εξεμεσ\w*).{0,20}(αιμα|αιματ)|αιματηρ|ματωμεν"),
         "Blood in vomit or diarrhoea can mean haemorrhagic gastroenteritis, parvovirus or an obstruction, and dehydration can set in fast: see a vet today.",
         "Αίμα στον εμετό ή στη διάρροια μπορεί να σημαίνει αιμορραγική γαστρεντερίτιδα, παρβοϊό ή απόφραξη και η αφυδάτωση έρχεται γρήγορα: δες κτηνίατρο σήμερα.",
         SRC["dog_gi"], "msd", exclude=r"rat poison|rodenticide|ποντικοφαρμακ"),
    Rule("DOG-PARVO", ("dog",), "EMERGENCY",
         _r(r"parvo|παρβο|(puppy|pup\b|κουταβ|\b\d+ ?(weeks?|εβδομαδ)).{0,60}(bloody|blood|αιμα|αιματ)",
            r"vomit|diarrh|lethar|bloody|blood|sick|εμετ|διαρρο|ληθαργ|αιμα"),
         "A puppy or unvaccinated dog with vomiting and bloody diarrhoea may have parvovirus, which can kill within hours without treatment: go to a vet now and keep them away from other dogs.",
         "Ένα κουτάβι ή αεμβολίαστος σκύλος με εμετό και αιματηρή διάρροια μπορεί να έχει παρβοϊό, που μπορεί να σκοτώσει μέσα σε ώρες χωρίς θεραπεία: πήγαινε τώρα σε κτηνίατρο και κράτα τον μακριά από άλλους σκύλους.",
         SRC["dog_gi"], "msd", exclude=r"vaccinated for parvo|εμβολιασμεν\w* για παρβο|parvo (shot|vaccine|vaccination)", immediate=True),
    Rule("DC-FOREIGN-BODY", ("dog", "cat"), "URGENT",
         _r(r"\bsock|stocking|\bstring\b|\byarn|\bthread|tinsel|needle|ribbon|hair ?tie|rubber band|corn ?cob|fishing (line|hook)|"
            r"καλτσα|κλωστη|βελονα|κορδελα|λαστιχακι|καλαμποκι.{0,8}(ροκα|στελεχοσ)|πετονι|αγκιστρι", _INGEST),
         "A swallowed sock, string, needle, ribbon or corn cob can block or cut the gut: call a vet today, do not pull anything hanging out, and go now if vomiting, belly pain or retching starts.",
         "Κάλτσα, κλωστή, βελόνα, κορδέλα ή ρόκα καλαμποκιού που καταπίθηκε μπορεί να φράξει ή να κόψει το έντερο: κάλεσε κτηνίατρο σήμερα, μην τραβήξεις κάτι που κρέμεται και πήγαινε αμέσως αν αρχίσει εμετός, πόνος ή ξερομάρισμα.",
         SRC["dog_gi"], "msd"),
    # ============ small mammals ("other": guinea pig, hamster, ferret, rat, mouse, chinchilla) from the MSD all-other-pets pages
    Rule("OTH-NOEAT", ("other",), "URGENT",
         _r(r"not eating|stopped eating|won'?t eat|no appetite|hasn'?t eaten|refus\w* (food|hay|pellet)|off (his|her) food|no food|"
            r"δεν τρωει|σταματησε να τρωει|δεν εφαγε|χωρις ορεξη|δεν θελει φαγητο"),
         "A small pet (guinea pig, hamster, ferret, rat) that stops eating needs a vet promptly: they dehydrate and decline fast, and some die suddenly.",
         "Ένα μικρό κατοικίδιο (ινδικό χοιρίδιο, χάμστερ, κουνάβι, αρουραίος) που σταματά να τρώει χρειάζεται κτηνίατρο άμεσα: αφυδατώνεται και χειροτερεύει γρήγορα και μερικά πεθαίνουν ξαφνικά.",
         SRC["gp_common"], "msd"),
    Rule("OTH-RESP", ("other",), "URGENT",
         _r(r"wheez|nasal discharge|runny nose|eye discharge|sneez|breathing (fast|heavily|hard|noisily)|noisy breathing|"
            r"συριγμ|ρινικ\w* εκκριση|εκκριση απο τη μυτη|φταρνισ|γρηγορη αναπνο|βαρια αναπνο"),
         "Wheezing, sneezing, nose or eye discharge or fast breathing in a small pet can be pneumonia, a leading cause of death in guinea pigs and hamsters: see a vet promptly.",
         "Συριγμός, φτέρνισμα, έκκριση από μύτη ή μάτια ή γρήγορη αναπνοή σε μικρό κατοικίδιο μπορεί να είναι πνευμονία, κύρια αιτία θανάτου στα ινδικά χοιρίδια και τα χάμστερ: δες κτηνίατρο άμεσα.",
         SRC["gp_common"] + " | " + SRC["hamster"], "msd"),
    Rule("OTH-DIARRHEA", ("other",), "URGENT",
         _r(r"diarrh|wet tail|watery (stool|poop)|runny (stool|poop)|διαρροια|υγρη ουρα|νερουλ\w* κοπρανα"),
         "Diarrhoea (wet tail) in a small pet is serious: dehydration and rapid decline are common, and giving antibiotics that were not prescribed can make it worse.",
         "Η διάρροια (υγρή ουρά) σε μικρό κατοικίδιο είναι σοβαρή: αφυδάτωση και γρήγορη επιδείνωση είναι συχνές και αντιβιοτικά που δεν έχουν συνταγογραφηθεί μπορούν να το χειροτερέψουν.",
         SRC["hamster"] + " | " + SRC["ferret_emerg"], "msd"),
    Rule("OTH-DIARRHEA-YOUNG", ("other",), "EMERGENCY",
         _r(r"diarrh|wet tail|watery (stool|poop)|διαρροια|υγρη ουρα|νερουλ\w* κοπρανα",
            r"baby|young|juvenile|weeks? old|just weaned|weaned|new ?born|μωρο|μικρο|νεαρο|εβδομαδ|απογαλακτ"),
         "Diarrhoea in a young hamster or other young small pet (proliferative ileitis, Tyzzer disease) can kill within a day or two: go to a vet now.",
         "Διάρροια σε νεαρό χάμστερ ή άλλο νεαρό μικρό κατοικίδιο (εντερίτιδα, νόσος Tyzzer) μπορεί να σκοτώσει μέσα σε μία με δύο μέρες: πήγαινε σε κτηνίατρο τώρα.",
         SRC["hamster"], "msd", immediate=True),
    Rule("OTH-PROLAPSE", ("other", "dog", "cat", "rabbit"), "EMERGENCY",
         _r(r"prolaps|(intestine|bowel|rectum|gut).{0,20}(sticking|protrud|hanging|coming) out|(sticking|protrud|hanging) out.{0,20}(bottom|anus|rectum)|"
            r"προπτωσ|εντερο.{0,15}εξω|προβαλλ\w*.{0,15}(πρωκτ|ορθο)"),
         "Intestine or rectum protruding from the bottom (prolapse) is an emergency needing immediate veterinary care.",
         "Έντερο ή ορθό που προβάλλει από τον πρωκτό (πρόπτωση) είναι επείγον και χρειάζεται άμεση κτηνιατρική φροντίδα.",
         SRC["hamster"] + " | " + SRC["ferret_emerg"], "msd", immediate=True),
    Rule("OTH-FERRET-NEURO", ("other",), "EMERGENCY",
         _r(r"ferret|κουναβι",
            r"seiz|convuls|staggering|dragging|can'?t walk|weak|lethar|collaps|pawing at (the |his |her )?mouth|drool|"
            r"σπασμ|τρικλιζ|σερν|δεν περπατ|αδυναμ|ληθαργ|σαλια"),
         "A ferret that is weak, staggering, dragging its back legs, drooling or having seizures may have low blood sugar (insulinoma) or another emergency: see a vet immediately.",
         "Ένα κουνάβι που είναι αδύναμο, τρικλίζει, σέρνει τα πίσω πόδια, σαλιώνει ή έχει σπασμούς μπορεί να έχει χαμηλό σάκχαρο (ινσουλίνωμα) ή άλλο επείγον: πήγαινε αμέσως σε κτηνίατρο.",
         SRC["ferret_emerg"], "msd", immediate=True),
    Rule("OTH-FERRET-GI", ("other",), "URGENT",
         _r(r"ferret|κουναβι",
            r"vomit|retch|stopped (eating|pooping|defecating)|no (poop|stool|droppings)|black (tarry )?stool|tarry|choking|cough|"
            r"εμετ|ξερατ|δεν κανει κοπρανα|μαυρα κοπρανα|βηχ"),
         "A ferret that vomits, stops eating or passing stool, or has black tarry stool may have a blockage or bleeding and dehydrates quickly: see a vet today.",
         "Ένα κουνάβι που κάνει εμετό, σταματά να τρώει ή να κάνει κόπρανα ή έχει μαύρα κόπρανα μπορεί να έχει απόφραξη ή αιμορραγία και αφυδατώνεται γρήγορα: δες κτηνίατρο σήμερα.",
         SRC["ferret_emerg"], "msd"),
    Rule("OTH-GP-SCURVY", ("other",), "URGENT",
         _r(r"guinea ?pig|cavy|χοιριδι\w* ινδι|ινδικ\w* χοιριδι",
            r"swollen (joint|leg|paw)|limping|lame|bleeding gums?|rough coat|bruis|πρηγμεν\w* (αρθρ|ποδ)|κουτσαιν|ματωνουν|ουλα.{0,10}αιμα|μελανιασ"),
         "Swollen joints, limping, bleeding gums or bruising in a guinea pig suggest vitamin C deficiency (scurvy), which can be fatal if untreated: see a vet promptly.",
         "Πρησμένες αρθρώσεις, κουτσαμάρα, αιμορραγία στα ούλα ή μελανιές σε ινδικό χοιρίδιο παραπέμπουν σε έλλειψη βιταμίνης C (σκορβούτο), που μπορεί να είναι θανατηφόρο χωρίς θεραπεία: δες κτηνίατρο άμεσα.",
         SRC["gp_nutrition"], "msd"),
    # ============ gaps found by running real MSD case-study presentations through the rules (policy: need vet sign-off)
    Rule("ALL-NEURO", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"head ?press|pressing (his|her|its|the) head|disorient|circling|walking in circles|stumbl|unsteady|ataxi|wobbl|staggering|can'?t walk straight|lost (his|her) balance|head tilt|"
            r"πιεζει το κεφαλι|κεφαλι.{0,12}(τοιχο|γωνια)|ζαλισμεν|γυριζει κυκλους|γυρναει κυκλους|τρικλιζ|χανει την ισορροπι|γερνει το κεφαλι"),
         "Head pressing, disorientation, circling, stumbling or a head tilt point to a brain or balance problem: see a vet today, and go now if it comes with seizures or collapse.",
         "Πίεση του κεφαλιού, αποπροσανατολισμός, κυκλικές κινήσεις, τρικλίσματα ή γερμένο κεφάλι παραπέμπουν σε πρόβλημα εγκεφάλου ή ισορροπίας: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν συνοδεύεται από σπασμούς ή κατάρρευση.",
         "", "policy"),
    Rule("CAT-ABDOMEN", ("cat",), "URGENT",
         _r(r"(swollen|distended|bloated|enlarged|big|fat)\w* .{0,10}(belly|abdomen|tummy|stomach)|(belly|abdomen|tummy|stomach).{0,20}(swollen|distended|bloated|enlarged)|φουσκωμεν\w* κοιλι|κοιλι\w*.{0,15}(πρησμεν|φουσκωμεν)",
            r"letharg|weight loss|losing weight|not eating|poor appetite|reduced appetite|no appetite|weak|ληθαργ|αδυνατισ|χασε βαρος|δεν τρωει|αδυναμ"),
         "A swollen belly with lethargy, weight loss or poor appetite in a cat can mean fluid (for example from FIP), organ disease or a tumour: see a vet within the day.",
         "Φουσκωμένη κοιλιά με λήθαργο, απώλεια βάρους ή κακή όρεξη σε γάτα μπορεί να σημαίνει υγρό (π.χ. από FIP), νόσο οργάνου ή όγκο: δες κτηνίατρο μέσα στην ημέρα.",
         SRC["cases"], "policy"),
    Rule("DC-GI-PERSISTENT", ("dog", "cat"), "URGENT",
         _r(r"vomit|throw(ing)? up|threw up|diarrh|εμετ|ξερασ|διαρρο",
            r"(for|since|over|last|past) .{0,12}(\b([2-9]|\d\d)\b ?(days?|μερ)|several days|few days|days|couple of days)|\b([2-9]|\d\d)\b ?(days?|μερες)|several days|few days|for days|μερες|εδω και μερες|3 με 4|3-4"),
         "Vomiting or diarrhoea lasting two days or more, especially with poor appetite or drinking, dehydrates a pet and can hide kidney, liver or gut disease: see a vet today.",
         "Εμετός ή διάρροια που κρατά δύο μέρες ή περισσότερο, ιδίως με κακή όρεξη ή χωρίς νερό, αφυδατώνει το ζώο και μπορεί να κρύβει νεφρική, ηπατική ή εντερική νόσο: δες κτηνίατρο σήμερα.",
         SRC["dog_gi"] + " | " + SRC["cases"], "policy",
         exclude=r"\bonce\b|one time|only once|just once|\bago\b|μια φορα|μονο μια|πριν απο"),
    Rule("DC-HEMATURIA", ("dog", "cat"), "URGENT",
         _r(r"blood in (his|her|the|their)? ?(urine|pee|wee)|bloody (urine|pee|wee)|(urine|pee|wee) .{0,12}(blood|bloody|red|pink)|red urine|pink urine|αιμα στα ουρα|αιματουρ|ουρα.{0,12}(αιμα|κοκκιν|ροζ)|κοκκινα ουρα"),
         "Blood in the urine can come from a bladder infection, stones, or a tumour, and a blockage can follow: see a vet today, and go now if there is straining with little or no urine.",
         "Αίμα στα ούρα μπορεί να οφείλεται σε κυστίτιδα, πέτρες ή όγκο και μπορεί να ακολουθήσει απόφραξη: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν υπάρχει προσπάθεια ούρησης με λίγα ή καθόλου ούρα.",
         SRC["cases"], "policy"),
    Rule("DC-LAME-SEVERE", ("dog", "cat"), "URGENT",
         _r(r"won'?t (put|bear) (any )?weight|not (putting|bearing) (any )?weight|can'?t (put|bear) (any )?weight|three[- ]legged|on three legs|holding (up )?(a |his |her |the )?(front |back |hind )?(leg|paw|limb)|carrying (a |his |her |the )?(leg|paw)|"
            r"yelp\w*.{0,40}(leg|limb|paw|jump|fell)|fell .{0,30}(then|and) .{0,20}(lame|limp)|severely lame|"
            r"δεν πατα|τρια ποδια|σηκωνει το ποδι|κρατα ψηλα το ποδι|κουτσαιν\w* πολυ|πολυ κουτσ|ουρλιαζ.{0,30}ποδι"),
         "Not putting weight on a leg, holding it up, or yelping with a sudden severe limp can mean a fracture, a torn ligament or a bone problem: see a vet today.",
         "Αν δεν πατά το πόδι, το κρατά ψηλά ή ουρλιάζει με ξαφνικό σοβαρό κούτσαμα, μπορεί να είναι κάταγμα, ρήξη συνδέσμου ή πρόβλημα στο οστό: δες κτηνίατρο σήμερα.",
         SRC["emergency"], "msd"),
    # ---- from the second batch of MSD case studies (M009-M021); policy basis unless noted: a vet must confirm
    Rule("DC-CANT-WALK", ("dog", "cat"), "EMERGENCY",
         _r(r"(can'?t|cannot|unable to|not able to|won'?t be able to|couldn'?t) (walk|stand|get up|use (his|her) (back|hind|rear) legs)|paraly[sz]|dragging (his|her) (back|hind|rear) (legs?|end)|"
            r"δεν μπορει (πλεον )?να (περπατησ|σηκωθ|σταθ)|ακινητοποιηθηκε|παραλυσ|σερνει τα (πισω )?ποδια"),
         "A pet that suddenly cannot walk or stand can have a spinal, nerve, heart or poisoning emergency: go to a vet now.",
         "Ζώο που ξαφνικά δεν μπορεί να περπατήσει ή να σταθεί μπορεί να έχει επείγον πρόβλημα σπονδυλικής στήλης, νεύρων, καρδιάς ή δηλητηρίαση: πήγαινε σε κτηνίατρο αμέσως.",
         SRC["cases"], "policy", immediate=True),
    Rule("DC-NOSEBLEED", ("dog", "cat", "rabbit"), "EMERGENCY",
         _r(r"nose ?bleed|epistaxis|blood (spray\w*|comes?|coming|pour\w*|drip\w*|runs?|running|squirt\w*) .{0,20}nose|bleed\w* (from|out of) (his|her|the|its) nose|blood from (his|her|the|its) nose|"
            r"αιμορραγι\w* απο (τη|την) μυτη|αιμα απο (τη|την) μυτη|μυτη.{0,15}(αιμορραγ|αιματ)"),
         "Heavy or repeated nosebleeds can mean a clotting problem, poisoning, a tumour or a nasal infection: go to a vet now.",
         "Έντονη ή επαναλαμβανόμενη ρινορραγία μπορεί να οφείλεται σε διαταραχή πήξης, δηλητηρίαση, όγκο ή λοίμωξη της μύτης: πήγαινε σε κτηνίατρο αμέσως.",
         SRC["cases"], "policy", immediate=True),
    Rule("DC-JAUNDICE", ("dog", "cat"), "URGENT",
         _r(r"yellow\w* (skin|gums?|eyes?|whites|ears?)|(skin|gums?|eyes?|whites|ears?).{0,20}yellow|jaundic|icter|"
            r"κιτρινο (δερμα|ματια|ουλα)|(δερμα|ματια|ουλα).{0,20}κιτριν|κιτρινισ|ικτερ"),
         "Yellow skin, gums or eyes (jaundice) means a liver, gallbladder or blood problem: see a vet today.",
         "Κίτρινο δέρμα, ούλα ή μάτια (ίκτερος) δείχνει πρόβλημα στο συκώτι, στη χολή ή στο αίμα: δες κτηνίατρο σήμερα.",
         SRC["cases"], "policy"),
    Rule("DC-ANOREXIA-LONG", ("dog", "cat"), "URGENT",
         _r(r"(not|n'?t|never|stopped|refus\w*) .{0,16}(eat|eaten|eating|food)\b|δεν (τρωει|εφαγε|εχει φαει|θελει να φαει|παιρνει φαγητο)|αρνειται (το )?φαγητο|ανορεξ",
            r"(for|since|over|more than|nearly|almost) .{0,10}(24 ?(hours|hrs|h)|a day|one day|two days|\b([2-9]|\d\d)\b ?days|a week|week|weeks)|a week|\d+ days|24 hours|μερες|εβδομαδ|προχθες|24 ωρες|μια μερα"),
         "A dog or cat that has not eaten for a day or more, especially with lethargy, vomiting or pain, should be seen by a vet today.",
         "Σκύλος ή γάτα που δεν έχει φάει για μία μέρα ή περισσότερο, ιδίως με ληθαργικότητα, εμετό ή πόνο, πρέπει να εξεταστεί από κτηνίατρο σήμερα.",
         SRC["cases"], "policy", exclude=r"\b(just|only) (a|one) (meal|bite)|fasting|fasted|surgery|before (the )?(op|operation|surgery)"),
    Rule("DC-WOUND-NONHEAL", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"not healing|won'?t heal|isn'?t healing|aren'?t healing|(wound|bite|bites|abscess|sore)s?.{0,50}(infected|pus|oozing|smell\w*|swollen|getting worse|worse)|"
            r"δεν γιατρευ|δεν κλεινει|πληγ\w*.{0,30}(πυο|μολυσμ|χειροτερ|βρωμ)"),
         "A wound or bite that is not healing, or is swollen, oozing or smelly, is infected until proven otherwise: see a vet today.",
         "Πληγή ή δάγκωμα που δεν επουλώνεται ή είναι πρησμένο, τρέχει ή μυρίζει θεωρείται μολυσμένο μέχρι να αποδειχθεί το αντίθετο: δες κτηνίατρο σήμερα.",
         SRC["cases"], "policy"),
    Rule("DC-LUMP-OPEN", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"(lumps?|swellings?|bumps?|masses|mass|growths?|tumou?rs?).{0,90}(burst|ruptur|\bopen\b|opened|oozing|draining|bleeding|discharg|ulcerat)|(burst|ruptured|opened).{0,20}(lumps?|swellings?|growths?)|"
            r"(εξογκωμα|εξογκωματα|κομποι|ογκο\w*|μαζα|μπαλιτσα).{0,40}(εσπασ|ανοιξ|αιμορραγ|τρεχ|πυο)"),
         "A lump that has opened, bleeds or oozes can be infected or a tumour: see a vet within the day.",
         "Εξόγκωμα που άνοιξε, αιμορραγεί ή τρέχει μπορεί να είναι μολυσμένο ή όγκος: δες κτηνίατρο μέσα στην ημέρα.",
         SRC["cases"], "policy"),
    Rule("DOG-URINARY-STRAIN", ("dog",), "URGENT",
         _r(r"strain\w* to (pee|urinate|wee)|strain\w* .{0,6}(pee|urinat|wee)|(pee|urinat|wee)\w* .{0,6}straining|trying to (pee|urinate|wee)|cannot (pee|urinate)|can'?t (pee|urinate)|"
            r"προσπαθει να ουρησει|δυσκολευεται να ουρησει|ουρα.{0,12}(σταγον|δυσκολ)"),
         "A dog straining to urinate can have stones, a blocked urethra, a prostate problem or a tumour: see a vet today, and go now if no urine comes out at all.",
         "Σκύλος που δυσκολεύεται να ουρήσει μπορεί να έχει πέτρες, απόφραξη ουρήθρας, πρόβλημα προστάτη ή όγκο: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν δεν βγαίνει καθόλου ούρο.",
         SRC["cases"], "policy"),
    # ---- from the MSD veterinary Emergency Medicine pages (policy basis: owner-level reading of clinician text)
    Rule("ALL-EYE-PENETRATING", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"eye.{0,30}(punctur|pierc|stabb|ruptur|burst|cut open|lacerat|gash)|(punctur|pierc|stabb|ruptur|lacerat)\w* .{0,15}eye|"
            r"ματι.{0,30}(τρυπ|σκισ|σπασ|χτυπ.{0,10}αιμ)|τρυπησε.{0,15}ματι"),
         "A punctured, cut or ruptured eye can be lost within hours: go to a vet now and do not press on the eye.",
         "Μάτι που τρυπήθηκε, κόπηκε ή έσπασε μπορεί να χαθεί μέσα σε ώρες: πήγαινε αμέσως σε κτηνίατρο και μην πιέζεις το μάτι.",
         MSD + "emergency-medicine-and-critical-care/ophthalmic-emergencies-in-small-animals/overview-of-ophthalmic-emergencies-in-small-animals", "policy", immediate=True),
    Rule("ALL-EYE-URGENT", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"(suddenly|sudden|overnight) .{0,12}(blind|can'?t see|lost (his|her) (sight|vision)|vision loss)|went blind|bumping into (things|furniture|walls)|"
            r"(eye|eyes?).{0,30}(cloudy|cloud|blue|hazy|bloodshot).{0,40}(pain|squint|red|closed|rubbing)|(red|cloudy|painful|squinting|swollen|bulging).{0,20}eye|eye.{0,20}(red and|swollen shut|squint|kept closed|closed shut)|"
            r"something (stuck|in) (in )?(his|her|the) eye|scratch\w* .{0,12}eye|eye.{0,15}scratch|"
            r"ξαφνικα (τυφλ|δεν βλεπ)|χτυπαει στα (πραγματα|επιπλα)|ματι.{0,25}(κοκκιν|θολο|πρησμεν|κλειστο|πονα)|(κοκκινο|θολο|πρησμενο) ματι"),
         "A red, cloudy, swollen or squinting eye, a scratch or foreign body in the eye, or sudden loss of vision can damage the eye permanently within a day or two: see a vet today.",
         "Κόκκινο, θολό, πρησμένο ή μισόκλειστο μάτι, γρατζουνιά ή ξένο σώμα στο μάτι ή ξαφνική απώλεια όρασης μπορεί να βλάψει μόνιμα το μάτι μέσα σε μια-δυο μέρες: δες κτηνίατρο σήμερα.",
         MSD + "emergency-medicine-and-critical-care/ophthalmic-emergencies-in-small-animals/overview-of-ophthalmic-emergencies-in-small-animals", "policy"),
    Rule("ALL-TRAUMA-MINOR", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"kicked|stepped on|trod on|dropped (him|her|the)|fell (down )?(the )?(stairs|steps)|fell (off|from) (the )?(couch|bed|sofa|table|counter|chair)|crushed|trapped (in|under)|slammed|caught in (a |the )?door|"
            r"κλωτσ|πατησε|του εφυγε απο τα χερια|επεσε (απο )?τις σκαλες|επεσε απο (τον|το|την) (καναπε|κρεβατι|τραπεζι|καρεκλα)|πιαστηκε στην πορτα|καταπλακωθ|πιεστηκε"),
         "Even a knock that looks minor can cause internal bleeding, chest or spine injury that shows up hours or days later: see a vet today.",
         "Ακόμη και ένα χτύπημα που φαίνεται μικρό μπορεί να προκαλέσει εσωτερική αιμορραγία ή τραυματισμό θώρακα ή σπονδυλικής που φαίνεται ώρες ή μέρες μετά: δες κτηνίατρο σήμερα.",
         MSD + "emergency-medicine-and-critical-care/specific-diagnostics-and-therapy/trauma-in-emergency-medicine-in-small-animals", "policy"),
    Rule("DC-BITE-NECK-SWELL", ("dog", "cat"), "EMERGENCY",
         _r(r"(bit|bite|bites|bitten|fight|attack\w*).{0,60}(neck|throat).{0,60}(swell|swollen|swelling|puffy)|(neck|throat).{0,40}(swell|swollen|swelling).{0,60}(bit|bite|bites|bitten|fight|attack)|"
            r"(δαγκ|καβγ|επιθεσ).{0,60}(λαιμ|τραχηλ).{0,60}(πρησ|πρηξ|φουσκ)|(λαιμ|τραχηλ).{0,40}(πρησ|πρηξ|φουσκ).{0,60}(δαγκ|καβγ|επιθεσ)"),
         "Swelling of the neck after a bite can squeeze the windpipe: go to a vet now.",
         "Πρήξιμο του λαιμού μετά από δάγκωμα μπορεί να πιέσει την τραχεία: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "emergency-medicine-and-critical-care/specific-diagnostics-and-therapy/trauma-in-emergency-medicine-in-small-animals", "policy", immediate=True),
    # ---- from the MSD veterinary urinary-system and musculoskeletal pages (policy basis; triage levels are inferred, MSD rarely states them)
    Rule("DOG-URETHRAL", ("dog",), "EMERGENCY",
         _r(r"strain\w*|squat\w*|trying to (pee|urinate|wee)|keeps? (going|trying)|προσπαθει να ουρησει|σκυβει",
            r"nothing (comes?|is coming|came) out|no urine|not producing (any )?urine|only (a |the )?(few |tiny )?(drop|dribble|bit)|just (a )?(drop|dribble)|can'?t (pee|urinate|wee)|cannot (pee|urinate)|unable to (pee|urinate)|"
            r"δεν (βγαινει|βγαζει|ουρει|καταφερνει)|σταγον|καθολου ουρ|τιποτα"),
         "A dog that strains but passes little or no urine may have a blocked urethra: poisons build up within 1 to 2 days, so go to a vet now.",
         "Σκύλος που σφίγγεται αλλά βγάζει λίγο ή καθόλου ούρο μπορεί να έχει απόφραξη ουρήθρας: οι τοξίνες συσσωρεύονται σε 1-2 μέρες, πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "urinary-system/urolithiasis-in-small-animals/urethral-obstruction-in-small-animals", "policy", immediate=True),
    Rule("DC-NO-URINE-DAY", ("dog", "cat"), "EMERGENCY",
         _r(r"(hasn'?t|has not|have not|haven'?t|didn'?t|did not|not|no)\b.{0,14}(peed|urinated|passed urine|weed|urine)|no urine|stopped (peeing|urinating)|δεν (εχει )?(κατουρησε|κατουραει|ουρησε|ουρει|βγαζει ουρα)|καθολου ουρα|σταματησε να (κατουραει|ουρει)",
            r"(all day|since (yesterday|last night|this morning|morning)|24 ?(hours|hrs|h)|12 ?(hours|hrs|h)|a whole day|(one|two|\d) days?|overnight|for hours|ολη μερα|απο χθες|απο το πρωι|απο το βραδυ|ολο το βραδυ|24 ωρες|12 ωρες|δυο μερες|\d+ μερες|εδω και ωρες|για ωρες)"),
         "No urine for half a day or more can mean a blockage or kidney failure: go to a vet now.",
         "Καθόλου ούρα για μισή μέρα ή περισσότερο μπορεί να σημαίνει απόφραξη ή νεφρική ανεπάρκεια: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "urinary-system/noninfectious-diseases-of-the-urinary-system-in-small-animals/renal-dysfunction-in-dogs-and-cats", "policy", immediate=True,
         exclude=r"vomit|εμετ"),
    Rule("DC-URINARY-SICK", ("dog", "cat"), "EMERGENCY",
         _r(r"strain\w* (to|when|while|trying to) (pee|urinate|wee)|blood in (his|her|the)? ?(urine|pee)|bloody (urine|pee)|(peeing|urinating) (a lot|often|frequently|outside)|frequent(ly)? (pee|urinat)|small amounts? of urine|"
            r"ουρα.{0,10}αιμα|αιματουρ|ουρει συχνα|προσπαθει να ουρησει",
            r"vomit\w*|threw up|throwing up|collaps\w*|very weak|too weak|can'?t stand|lethargic and|εμετ|ξερασ|κατερρευσε|πολυ αδυναμ"),
         "Urinary signs together with vomiting or weakness suggest a blockage or kidney injury that is building up toxins: go to a vet now.",
         "Ουρολογικά συμπτώματα (δυσκολία ή συχνή ούρηση, αίμα στα ούρα) μαζί με εμετό ή αδυναμία δείχνουν απόφραξη ή νεφρική βλάβη: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "urinary-system/noninfectious-diseases-of-the-urinary-system-in-small-animals/obstructive-uropathy-in-dogs-and-cats", "policy", immediate=True),
    Rule("CAT-LUTD", ("cat",), "URGENT",
         _r(r"(pee\w*|urinat\w*|wee\w*) (outside|beside|next to|on the (floor|carpet|bed|sofa|couch|bath|sink|tiles?))|outside (of )?the (litter|box|tray)|frequent(ly)? (pee|urinat)|small amounts? of urine|many (small )?(pees|urinations)|licking (his|her) (genitals?|private)|"
            r"ουρει εξω απο|εξω απο την (αμμο|λεκανη)|ουρει (συχνα|λιγο λιγο)|γλειφει τα γεννητικα"),
         "A cat that pees outside the litter box, very often, or in tiny amounts can have cystitis, stones or an early blockage: see a vet today, and go now if nothing comes out.",
         "Γάτα που ουρεί εκτός αμμοδοχείου, πολύ συχνά ή σε μικρές ποσότητες μπορεί να έχει κυστίτιδα, πέτρες ή αρχόμενη απόφραξη: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν δεν βγαίνει τίποτα.",
         MSD + "urinary-system/noninfectious-diseases-of-the-urinary-system-in-small-animals/lower-urinary-tract-disease-in-cats", "policy"),
    Rule("DC-FRACTURE-OPEN", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"bone (is )?(sticking|poking|showing|exposed|visible|protruding)|open fracture|compound fracture|protruding bone|"
            r"κοκκαλο.{0,15}(βγαινει|φαινεται|εξω)|ανοιχτο κατα[γκ]μα"),
         "A bone showing through the skin is an emergency: cover the wound with a clean cloth, keep the pet still and go to a vet now.",
         "Οστό που φαίνεται έξω από το δέρμα είναι επείγον: κάλυψε την πληγή με καθαρό ύφασμα, κράτα το ζώο ακίνητο και πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "musculoskeletal-system/osteopathies-in-small-animals/bone-fractures-in-dogs-and-cats", "policy", immediate=True),
    Rule("DC-FRACTURE", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"broken (leg|bone|paw|limb|hip|jaw|tail|pelvis)|fracture|(leg|limb|paw).{0,20}(dangling|hanging|bent (the )?wrong|at an odd angle|deformed|twisted|crooked|swinging)|hip (is )?(dislocat|popped out|out of)|dislocat|"
            r"σπασμεν\w* (ποδι|κοκκαλο|σαγονι|λεκανη)|κατα[γκ]μα|(ποδι|πατουσα).{0,20}(κρεμεται|στραβο|παραμορφ|ξεβγαλ)|εξαρθρ"),
         "A suspected broken bone or dislocated joint needs a vet today, because delay can turn a simple fracture into a complicated one: keep the pet still.",
         "Ύποπτο σπασμένο κόκαλο ή εξάρθρημα χρειάζεται κτηνίατρο σήμερα, γιατί η καθυστέρηση μπορεί να περιπλέξει ένα απλό κάταγμα: κράτα το ζώο ακίνητο.",
         MSD + "musculoskeletal-system/osteopathies-in-small-animals/bone-fractures-in-dogs-and-cats", "policy"),
    Rule("DC-JOINT-ACUTE", ("dog", "cat"), "URGENT",
         _r(r"(hot|warm|swollen|red|puffy)\w* .{0,12}(joint|knee|elbow|hock|wrist|ankle|shoulder)|(joint|knee|elbow|hock|wrist|ankle|shoulder).{0,25}(hot|warm|swollen|puffy)|"
            r"(πρησμεν|ζεστ|κοκκιν)\w* .{0,12}(αρθρωση|γονατο|αγκωνα|καρπο)|(αρθρωση|γονατο|αγκωνα).{0,20}(πρησ|ζεστ)",
            r"sudden\w*|today|yesterday|overnight|this morning|last night|fever|letharg|not eating|since (yesterday|last night)|ξαφνικα|σημερα|χθες|πυρετ|ληθαργ"),
         "A hot, swollen joint with sudden lameness, fever or lethargy can be a joint infection that destroys the joint if treated late: see a vet today.",
         "Ζεστή, πρησμένη άρθρωση με ξαφνικό κούτσαμα, πυρετό ή ληθαργικότητα μπορεί να είναι λοίμωξη της άρθρωσης που την καταστρέφει αν καθυστερήσει η θεραπεία: δες κτηνίατρο σήμερα.",
         MSD + "musculoskeletal-system/arthropathies-and-related-disorders-in-small-animals/septic-arthritis-in-dogs-and-cats", "policy",
         exclude=r"\b(a|one|two|three|several|\d+) (months?|weeks?)\b|for (weeks|months|a while)|μηνα|εβδομαδ"),
    Rule("DC-STIFF-HOT", ("dog", "cat"), "EMERGENCY",
         _r(r"(stiff|rigid|locked)\w* .{0,15}(leg|limb|muscle|body)|(leg|limb|muscle)s? .{0,12}(stiff|rigid|locked)|σκληρα? (ποδια|μυες)|ακαμπτ",
            r"dark (urine|pee)|brown (urine|pee)|(urine|pee).{0,12}(dark|brown)|coffee.{0,8}(urine|pee)|fever|very hot|overheat|panting (hard|heavily)|racing heart|fast breathing|σκουρα ουρα|υψηλ\w* πυρετ|λαχανιαζ"),
         "Rigid muscles with fever, fast breathing or dark urine can be malignant hyperthermia or muscle breakdown, which can stop the heart: go to a vet now.",
         "Σκληροί μύες με πυρετό, γρήγορη αναπνοή ή σκούρα ούρα μπορεί να είναι κακοήθης υπερθερμία ή διάσπαση μυών, που μπορεί να σταματήσει την καρδιά: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "musculoskeletal-system/myopathies-in-dogs-and-cats/malignant-hyperthermia-in-dogs-and-cats", "policy", immediate=True),
    Rule("CAT-WEAK-NECK", ("cat",), "URGENT",
         _r(r"neck (is )?(droop\w*|hanging|dropped|down|bent down)|head (is )?(droop\w*|hanging|held low|dropped)|can'?t lift (his|her) head|ventroflex|walking stiff and weak|suddenly weak|"
            r"κρεμαει το κεφαλι|δεν σηκωνει το κεφαλι|ξαφνικη αδυναμια"),
         "A cat with a drooping neck or sudden weakness can have dangerously low potassium: see a vet today, and go now if breathing is laboured or she cannot stand.",
         "Γάτα με πεσμένο λαιμό ή ξαφνική αδυναμία μπορεί να έχει επικίνδυνα χαμηλό κάλιο: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν αναπνέει με δυσκολία ή δεν στέκεται.",
         MSD + "musculoskeletal-system/myopathies-in-dogs-and-cats/feline-hypokalemic-polymyopathy", "policy"),
    # ---- from the MSD veterinary Toxicology pages (policy basis: levels inferred from treatment guidance, vet to confirm)
    Rule("DOG-MACADAMIA", ("dog",), "URGENT",
         _r(r"macadamia|μακαντα?μια|μακαδαμια", _INGEST),
         "Macadamia nuts can cause vomiting, weakness, a wobbly gait, tremors and fever within about 12 hours: call a vet today even if your dog seems fine.",
         "Τα μακαντάμια μπορούν να προκαλέσουν εμετό, αδυναμία, ασταθές βάδισμα, τρέμουλο και πυρετό μέσα σε περίπου 12 ώρες: κάλεσε κτηνίατρο σήμερα ακόμη κι αν ο σκύλος φαίνεται καλά.",
         MSD + "toxicology/food-hazards/macadamia-nut-toxicosis-in-dogs", "policy"),
    Rule("DC-ONION-GARLIC", ("dog", "cat"), "URGENT",
         _r(r"\bonions?\b|\bgarlic\b|\bchives?\b|\bleeks?\b|onion (powder|soup|rings)|κρεμμυδ|σκορδ|πρασ(ο|α)\b|μπουκοβο", _INGEST),
         "Onion, garlic, chives and leeks damage red blood cells, cats most of all: weakness, pale gums or dark urine show up days later, so call a vet today after any real amount.",
         "Το κρεμμύδι, το σκόρδο, το σχοινόπρασο και το πράσο καταστρέφουν τα ερυθρά αιμοσφαίρια, ιδίως στις γάτες: αδυναμία, ωχρά ούλα ή σκούρα ούρα εμφανίζονται μέρες μετά, γι' αυτό κάλεσε κτηνίατρο σήμερα αν έφαγε αξιόλογη ποσότητα.",
         MSD + "toxicology/food-hazards/garlic-and-onion-allium-spp-toxicosis-in-animals", "policy"),
    Rule("DC-MOTHBALL", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"moth ?balls?|naphthalene|camphor|paradichlorobenzene|ναφθαλιν|καμφορ|σκορακι\w* για (τους )?σκουληκ|αντισκωρικ", _INGEST),
         "Mothballs (naphthalene, camphor) damage blood and liver and can cause vomiting, tremors or seizures: call a vet now and do not make your pet vomit.",
         "Η ναφθαλίνη και η καμφορά βλάπτουν το αίμα και το συκώτι και μπορούν να προκαλέσουν εμετό, τρέμουλο ή σπασμούς: κάλεσε κτηνίατρο αμέσως και μην προκαλέσεις εμετό.",
         MSD + "toxicology/toxicoses-from-household-hazards/toxicosis-from-mothballs-in-animals", "policy"),
    Rule("CAT-ESSENTIAL-OIL", ("cat", "bird", "rabbit", "other"), "EMERGENCY",
         _r(r"essential oils?|tea ?tree|melaleuca|pennyroyal|wintergreen|eucalyptus oil|cinnamon oil|peppermint oil|citrus oil|oil diffuser|diffuser|αιθεριο \w*|αιθεριων ελαιων|τι τρι|δενδρολιβαν|ευκαλυπτ|διαχυτη",
            r"licked|swallow\w*|ingest\w*|spilled|spilt|applied|rubbed|on (his|her|their) (skin|fur|coat|back)|drank|drunk|εγλειψε|ηπιε|χυθηκε|εβαλα|πανω στο|στο τριχωμα|εισεπνευσε|breath\w* (it|them) in"),
         "Essential oils (tea tree, wintergreen, pennyroyal, eucalyptus and others) are toxic to cats and birds by skin, mouth or inhaled mist: wipe nothing off with water alone, keep the pet calm and call a vet now.",
         "Τα αιθέρια έλαια (τεϊόδεντρο, γκοϊλθέρια, ευκάλυπτος κ.ά.) είναι τοξικά για γάτες και πτηνά από το δέρμα, το στόμα ή την εισπνοή: κράτα το ζώο ήρεμο και κάλεσε κτηνίατρο αμέσως.",
         MSD + "toxicology/toxicoses-from-household-hazards/toxicoses-from-essential-oils-in-animals", "policy", immediate=True),
    Rule("DOG-ESSENTIAL-OIL", ("dog",), "URGENT",
         _r(r"essential oils?|tea ?tree|melaleuca|pennyroyal|wintergreen|eucalyptus oil|cinnamon oil|αιθεριο \w*|αιθεριων ελαιων|τι τρι|δενδρολιβαν",
            r"licked|swallow\w*|ingest\w*|spilled|spilt|drank|drunk|εγλειψε|ηπιε|χυθηκε"),
         "Essential oils swallowed or licked by a dog can cause vomiting, drooling, wobbliness and tremors: call a vet today.",
         "Τα αιθέρια έλαια που καταπίνει ή γλείφει ο σκύλος μπορούν να προκαλέσουν εμετό, σάλια, ασταθές βάδισμα και τρέμουλο: κάλεσε κτηνίατρο σήμερα.",
         MSD + "toxicology/toxicoses-from-household-hazards/toxicoses-from-essential-oils-in-animals", "policy"),
    Rule("DC-RODENTICIDE-NEURO", ("dog", "cat"), "EMERGENCY",
         _r(r"bromethalin|cholecalciferol|vitamin d3? (bait|rat|mouse|poison)|strychnine|chloralose|zinc phosphide|aluminium phosphide|aluminum phosphide|phosphide|"
            r"βρομεθαλιν|χολεκαλσιφερολ|στρυχνιν|χλωραλοζ|φωσφιδι", _INGEST),
         "Some rat poisons (bromethalin, vitamin D3, strychnine, chloralose, phosphides) attack the brain, nerves, kidneys or heart within hours or days even if the pet looks well: go to a vet now.",
         "Μερικά ποντικοφάρμακα (βρομεθαλίνη, βιταμίνη D3, στρυχνίνη, χλωραλόζη, φωσφίδια) επιτίθενται στον εγκέφαλο, τα νεύρα, τα νεφρά ή την καρδιά μέσα σε ώρες ή μέρες ακόμη κι αν το ζώο φαίνεται καλά: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "toxicology/rodenticide-poisoning/overview-of-rodenticide-poisoning-in-animals", "policy", immediate=True),
    Rule("DC-THC", ("dog", "cat"), "URGENT",
         _r(r"marijuana|marihuana|cannabis|hashish|\bthc\b|pot brownie|space cake|weed (brownie|cookie|gummy|gummies|edible)|edibles?\b.{0,20}(thc|cannabis|weed)|joint|μαριχουανα|κανναβ|χασισ|χασις",
            _INGEST + r"|smok\w*|inhal\w*|εφαγε|καπνισε|εισεπνευσε"),
         "Cannabis and THC edibles make pets wobbly, sleepy, sick or agitated for 1 to 3 days and can cause seizures or collapse: call a vet now.",
         "Η κάνναβη και τα προϊόντα THC κάνουν τα ζώα ασταθή, νυσταγμένα, ναυτιασμένα ή ανήσυχα για 1 έως 3 μέρες και μπορεί να προκαλέσουν σπασμούς ή κατάρρευση: κάλεσε κτηνίατρο αμέσως.",
         MSD + "toxicology/toxicoses-from-human-recreational-drugs/toxicosis-in-dogs-and-cats-from-tetrahydrocannabinol-thc", "policy"),
    Rule("DC-TOPICAL-HUMAN", ("dog", "cat"), "URGENT",
         _r(r"fentanyl|nicotine|cigarette|cigar\b|tobacco|vape|e-?liquid|nicotine (gum|patch|pouch)|minoxidil|rogaine|lidocaine|estrogen (patch|cream)|oestrogen|diclofenac|voltaren|voltarol|ibuprofen gel|"
            r"νικοτιν|τσιγαρ|καπνο\b|βολταρεν|μινοξιδιλ|λιδοκαιν|φεντανυλ|οιστρογον", _INGEST + r"|licked|applied|ate|patch|εγλειψε|εφαγε"),
         "Human patches, gels, nicotine and tobacco products can cause seizures, heart rhythm changes, kidney injury or a deadly drop in breathing in pets: call a vet now.",
         "Τα ανθρώπινα επιθέματα, οι γέλες, η νικοτίνη και τα προϊόντα καπνού μπορούν να προκαλέσουν σπασμούς, αρρυθμία, νεφρική βλάβη ή επικίνδυνη καταστολή της αναπνοής: κάλεσε κτηνίατρο αμέσως.",
         MSD + "toxicology/toxicosis-from-human-topical-agents/toxicosis-in-animals-from-human-topical-agents", "policy"),
    Rule("DC-CLEANER-POD", ("dog", "cat", "bird", "rabbit", "other"), "URGENT",
         _r(r"laundry (pod|capsule|detergent|liquid)|detergent (pod|capsule)|dishwasher (pod|tablet|capsule)|fabric softener|dish ?soap|washing (powder|liquid)|"
            r"απορρυπαντικ|κ(α|ά)ψουλ\w* (πλυντηριου|πλυσιματος)|μαλακτικ|υγρο πιατων|σκονη πλυντηριου", _INGEST + r"|bit (into|a|the|on)|bitten (into|a|the)|punctured|δαγκωσε|τρυπησε"),
         "Laundry and dishwasher pods and strong cleaners burn the mouth and gullet and cause repeated vomiting and breathing trouble: call a vet now and do not make your pet vomit.",
         "Οι κάψουλες πλυντηρίου/πιάτων και τα ισχυρά καθαριστικά καίνε το στόμα και τον οισοφάγο και προκαλούν επανειλημμένους εμετούς και δυσκολία στην αναπνοή: κάλεσε κτηνίατρο αμέσως και μην προκαλέσεις εμετό.",
         MSD + "toxicology/toxicoses-from-household-hazards/toxicoses-from-household-cleaners-and-personal-care-products-in-animals", "policy"),
    Rule("ALL-SMOKE", (), "EMERGENCY",
         _r(r"smoke inhalation|inhaled (a lot of )?smoke|breath\w* in smoke|breathed (in )?smoke|house fire|apartment fire|flat fire|barn fire|fire in the (house|apartment|flat|barn|kitchen|building)|rescued from (a |the )?fire|trapped in (a |the )?fire|covered in soot|soot (on|around|in) (his|her|the|its) (nose|mouth|face|nostrils)|"
            r"teflon|ptfe|non-?stick (pan|pot|pans)?.{0,40}(burn\w*|overheat\w*|smok\w*|fumes)|"
            r"εισεπνευσε καπνο|φωτια στο (σπιτι|διαμερισμα|κουζινα)|πυρκαγια|εμεινε μεσα σε φωτια|καπνισε το (σπιτι|διαμερισμα|κουζινα)|γεμισε καπνο|μεσα σε καπνο|τεφλον"),
         "Smoke, soot and overheated non-stick pans are dangerous even if the pet seems fine: lung swelling peaks 12 to 24 hours later, so go to a vet now (birds can die within minutes of Teflon fumes).",
         "Ο καπνός, η αιθάλη και τα υπερθερμασμένα αντικολλητικά σκεύη είναι επικίνδυνα ακόμη κι αν το ζώο φαίνεται καλά: το πνευμονικό οίδημα κορυφώνεται 12 έως 24 ώρες μετά, πήγαινε αμέσως σε κτηνίατρο (τα πτηνά μπορεί να πεθάνουν μέσα σε λεπτά από αναθυμιάσεις Teflon).",
         MSD + "toxicology/smoke-inhalation-injury/smoke-inhalation-injury-in-animals", "policy", immediate=True),
    # ---- from the MSD quiz check (Initial Triage quiz): gaps found when the quiz facts were compared with the rules
    Rule("ALL-NOISY-BREATH", ("dog", "cat", "rabbit"), "URGENT",
         _r(r"stridor|stertor|noisy breath\w*|breath\w* (is |was )?(noisy|loud|raspy|rasping|whistl\w*|wheez\w*)|wheez\w*|rasping|raspy breath\w*|high[- ]pitched (breath|sound|noise)|honking (cough|breath)|"
            r"θορυβωδ\w* αναπνο|σφυριγμ\w*.{0,20}αναπνο|αναπνο\w*.{0,15}(σφυριζ|βραχν|ρογχαλ)|ρογχαλ"),
         "Breathing that is loud, whistling, raspy or high-pitched means the airway is partly blocked or the lungs are struggling: see a vet today, and go now if breathing is hard, gums are pale or blue, or it is getting worse.",
         "Αναπνοή δυνατή, σφυριχτή, βραχνή ή οξεία σημαίνει μερική απόφραξη του αεραγωγού ή πρόβλημα στους πνεύμονες: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν η αναπνοή είναι δύσκολη, τα ούλα χλωμά ή μπλε ή χειροτερεύει.",
         MSD + "emergency-medicine-and-critical-care/evaluation-and-initial-treatment-of-small-animal-emergency-patients/initial-triage-and-resuscitation-of-small-animal-emergency-patients", "policy"),
    Rule("DC-GI-PROFUSE", ("dog", "cat"), "URGENT",
         _r(r"vomit\w*|threw up|throwing up|εμετ|ξερασ",
            r"diarrh\w*|loose stool|runny stool|διαρρο",
            r"(a lot|profuse\w*|repeated\w*|non-?stop|constant\w*|again and again|many times|multiple times|over and over|\b([4-9]|\d\d) times)|πολλ\w* φορ|συνεχει|ασταματητ|ξανα και ξανα"),
         "Heavy vomiting together with diarrhoea loses fluid fast and can become an emergency within hours: see a vet today.",
         "Έντονος εμετός μαζί με διάρροια χάνει υγρά γρήγορα και μπορεί να γίνει επείγον μέσα σε ώρες: δες κτηνίατρο σήμερα.",
         MSD + "emergency-medicine-and-critical-care/evaluation-and-initial-treatment-of-small-animal-emergency-patients/initial-triage-and-resuscitation-of-small-animal-emergency-patients", "policy"),
    # ---- from the full MSD quiz sweep (94 quizzes, 113 small-animal questions): gaps found against the rules; policy basis, vet to confirm
    Rule("DOG-ECLAMPSIA", ("dog", "cat"), "EMERGENCY",
         _r(r"nursing|lactating|whelp\w*|queen\w*|gave birth|giving birth|new puppies|newborn puppies|her (puppies|pups|kittens)|postpartum|post-partum|breastfeeding|feeding (her )?(puppies|pups|kittens)|"
            r"θηλαζ|γεννησε|τα κουταβια της|τα γατακια της|λοχεια|μετα τον τοκετο",
            r"tremor|trembl|shak\w*|twitch|stiff|rigid|panting|restless|pacing|wobbl|stagger|seiz|convuls|τρεμ|σπασμ|ακαμπτ|λαχανιαζ|ανησυχ|ασταθ"),
         "A mother dog or cat that is nursing and starts shaking, twitching, panting, stiff or restless may have dangerously low blood calcium (eclampsia): go to a vet now.",
         "Σκύλα ή γάτα που θηλάζει και αρχίζει να τρέμει, να σπαρταράει, να λαχανιάζει ή να είναι δύσκαμπτη ή ανήσυχη μπορεί να έχει επικίνδυνα χαμηλό ασβέστιο (εκλαμψία): πήγαινε αμέσως σε κτηνίατρο.",
         "MSD quizzes: Disorders of Calcium Metabolism (puerperal hypocalcemia)", "policy", immediate=True),
    Rule("CAT-FAST-BREATH", ("cat",), "URGENT",
         _r(r"breath\w* (very |really |too |so )?(fast|rapid|quick|heavy|hard|laboured|labored)|(fast|rapid|quick|heavy|laboured|labored) breath\w*|breathing (more|faster)|(chest|belly|sides|flanks?) .{0,12}(heaving|moving a lot|pumping)|"
            r"αναπνε\w* (γρηγορα|βαρια|με δυσκολια|πολυ)|γρηγορη αναπνοη|βαρια αναπνοη"),
         "Cats hide breathing trouble until it is severe; fast or heavy breathing at rest can be heart or lung disease: see a vet today, and go now if the mouth is open or the gums are pale or blue.",
         "Οι γάτες κρύβουν τη δυσκολία στην αναπνοή μέχρι να γίνει σοβαρή· γρήγορη ή βαριά αναπνοή σε ηρεμία μπορεί να είναι καρδιακή ή πνευμονική νόσος: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν το στόμα είναι ανοιχτό ή τα ούλα χλωμά ή μπλε.",
         "MSD quizzes: Heart Disease and Heart Failure (cats)", "policy"),
    Rule("DOG-HONK-COUGH", ("dog",), "URGENT",
         _r(r"goose[- ]?honk|honk\w*|dry hacking cough|hacking cough|cough\w* (at night|when excited|when (he|she) pulls|on the lead|on the leash)|coughing (a lot|more and more|every day)|"
            r"βηχει (τη νυχτα|οταν ενθουσιαζεται|πολυ|καθε μερα)|βηχας.{0,15}(χηνα|γαργαλ)"),
         "A honking or worsening cough can come from a collapsing windpipe or heart failure: see a vet within the day, and go now if breathing is laboured or the gums are blue.",
         "Βήχας που ακούγεται σαν κορνάρισμα ή χειροτερεύει μπορεί να οφείλεται σε κατάρρευση τραχείας ή καρδιακή ανεπάρκεια: δες κτηνίατρο μέσα στην ημέρα και πήγαινε αμέσως αν η αναπνοή είναι δύσκολη ή τα ούλα μπλε.",
         "MSD quizzes: Respiratory Diseases of Small Animals (tracheal collapse)", "policy"),
    Rule("DC-VOMIT-WATER", ("dog", "cat"), "URGENT",
         _r(r"(can'?t|cannot|won'?t|unable to|doesn'?t|does not|not able to|couldn'?t) keep (any )?(water|food|anything|fluids?) down|vomit\w* (everything|even water|after (every|each|any)|water)|throws? up (everything|even water|water)|"
            r"(even|and) (the )?water.{0,20}(vomit|comes up|throws)|ξερναει (ολα|και το νερο|νερο)|δεν κρατα (το )?(νερο|φαγητο|τιποτα)"),
         "A pet that vomits even water cannot stay hydrated and may have a blockage or serious illness: see a vet today.",
         "Ζώο που κάνει εμετό ακόμη και με το νερό δεν μπορεί να ενυδατωθεί και μπορεί να έχει απόφραξη ή σοβαρή νόσο: δες κτηνίατρο σήμερα.",
         "MSD quizzes: Various Obstructions (intussusception, intractable vomiting)", "policy"),
    Rule("DC-PUPD-WEIGHT", ("dog", "cat"), "URGENT",
         _r(r"drink\w* (much |a lot |way |far )?more|drinks? (a lot|too much|constantly)|always thirsty|very thirsty|empties (the |his |her )?(water )?bowl|(peeing|urinating|pees|urinates) (much |a lot |way |far )?more|\bpolydips|\bpolyuri|"
            r"πινει (πολυ )?περισσοτερο|πινει πολυ|διψα (παρα πολυ|παντα)|ουρει (πολυ )?περισσοτερο|ουρει πολυ",
            r"losing weight|lost weight|weight loss|thin(ner)?\b|eating (much |a lot |way )?more|always hungry|ravenous|begging (for )?food|constantly hungry|voracious|"
            r"χανει βαρος|εχασε βαρος|αδυνατισ|τρωει (πολυ )?περισσοτερο|πεινα(ει)? παντα|λαιμαργ"),
         "Drinking, peeing or eating much more together with weight loss can mean diabetes, kidney disease, Cushing's disease or an overactive thyroid: see a vet within a day or two, sooner if vomiting or very tired.",
         "Πολύ περισσότερο νερό, ούρα ή φαγητό μαζί με απώλεια βάρους μπορεί να σημαίνει διαβήτη, νεφρική νόσο, Cushing ή υπερθυρεοειδισμό: δες κτηνίατρο μέσα σε μία-δυο μέρες, νωρίτερα αν υπάρχει εμετός ή μεγάλη κούραση.",
         "MSD quizzes: Diabetes Mellitus in Dogs and Cats; The Thyroid Gland; Adrenal Glands", "policy"),
    Rule("CAT-RAVENOUS-THIN", ("cat",), "URGENT",
         _r(r"ravenous|always hungry|constantly hungry|eating (much |a lot |way )?more|voracious|πεινα(ει)? παντα|τρωει (πολυ )?περισσοτερο|λαιμαργ",
            r"losing weight|lost weight|weight loss|thin(ner)?\b|χανει βαρος|εχασε βαρος|αδυνατισ|ζυγιζει λιγοτερο"),
         "An older cat that eats a lot but loses weight often has an overactive thyroid, diabetes or a gut problem: see a vet within a day or two.",
         "Γάτα μεγάλης ηλικίας που τρώει πολύ αλλά χάνει βάρος συχνά έχει υπερθυρεοειδισμό, διαβήτη ή εντερικό πρόβλημα: δες κτηνίατρο μέσα σε μία-δυο μέρες.",
         "MSD quizzes: The Thyroid Gland (hyperthyroid cats)", "policy"),
    Rule("DOG-NECK-PAIN", ("dog",), "URGENT",
         _r(r"(neck|back|spine).{0,20}(pain\w*|rigid|stiff|hurts?)|yelp\w* (when|if) .{0,25}(touch\w*|pick\w*|lift\w*|mov\w*|head|neck)|(won'?t|doesn'?t|refuses to|reluctant to|afraid to) (move|turn|lower|lift) (his|her|the) (head|neck)|head held (low|down)|hunched|"
            r"πονα (ο |το )?(λαιμος|σπονδυλικη|πλατη)|ουρλιαζει οταν (τον|την) (πιανω|σηκωνω|ακουμπαω)|δεν κουναει (το )?κεφαλι|δεν γυριζει το κεφαλι|καμπουριασμεν"),
         "Neck or back pain with a stiff, hunched or head-low posture, yelping when touched or fever can mean a slipped disc or meningitis: see a vet today, and go now if the legs are weak or he cannot walk.",
         "Πόνος στον λαιμό ή στην πλάτη με δύσκαμπτη, καμπουριασμένη στάση ή χαμηλό κεφάλι, ουρλιαχτό στο άγγιγμα ή πυρετό μπορεί να σημαίνει κήλη δίσκου ή μηνιγγίτιδα: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν τα πόδια είναι αδύναμα ή δεν περπατά.",
         "MSD quizzes: Excessive Innate Responses (immune-mediated meningitis); Diseases of the Spinal Column and Cord", "policy"),
    # ---- from the MSD veterinary Reproductive System pages (dogs and cats; policy basis, levels inferred, vet to confirm)
    Rule("DC-PYOMETRA", ("dog", "cat"), "EMERGENCY",
         _r(r"un-?spayed|not spayed|intact (female|bitch|queen)|entire (female|bitch)|hasn'?t been spayed|has not been spayed|never been spayed|not been spayed|had a heat|in heat|season|after (her|a) heat|"
            r"δεν (εχει )?στειρωθει|μη στειρωμεν|ασπαγιαστ|ζευγαρωσ|οιστρο|ζεστη",
            r"discharge|pus|vulva|vaginal|drinking (much |a lot )?more|drinks (a lot|too much)|belly (is )?(swollen|bloated|big)|swollen belly|εκκριμα|πυο|κολπικ|πινει (πολυ|περισσοτερο)|φουσκωμενη κοιλια",
            r"letharg|not eating|won'?t eat|no appetite|vomit|fever|weak|collaps|very tired|ληθαργ|δεν τρωει|εμετ|πυρετ|αδυναμ"),
         "A sick, unspayed female (especially in the weeks after a heat) with discharge, extra thirst or a swollen belly may have a womb infection (pyometra), which can rupture: go to a vet now.",
         "Θηλυκό που δεν έχει στειρωθεί και είναι άρρωστο (ιδίως τις εβδομάδες μετά τον οίστρο) με έκκριμα, αυξημένη δίψα ή φουσκωμένη κοιλιά μπορεί να έχει πυομήτρα, που μπορεί να σπάσει: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "reproductive-system/female-reproductive-tract-disorders-in-small-animals/pyometra-in-small-animals", "policy", immediate=True),
    Rule("DC-PYOMETRA-DISCHARGE", ("dog", "cat"), "URGENT",
         _r(r"un-?spayed|not spayed|intact (female|bitch|queen)|entire (female|bitch)|hasn'?t been spayed|has not been spayed|never been spayed|not been spayed|"
            r"δεν (εχει )?στειρωθει|μη στειρωμεν|ασπαγιαστ",
            r"(pus|pus-like|foul|smelly|stinking|yellow\w*|green\w*|brown\w*|white|creamy|milky|cloudy) .{0,12}discharge|discharge .{0,20}(pus|foul|smell\w*|yellow|green|brown|white|creamy)|πυωδες εκκριμα|βρωμερο εκκριμα|εκκριμα.{0,15}(πυο|βρωμ|κιτριν|πρασιν)"),
         "Pus-like or foul discharge from an unspayed female can be a womb infection (pyometra): see a vet today.",
         "Πυώδες ή δυσάρεστης οσμής έκκριμα σε θηλυκό που δεν έχει στειρωθεί μπορεί να είναι πυομήτρα: δες κτηνίατρο σήμερα.",
         MSD + "reproductive-system/female-reproductive-tract-disorders-in-small-animals/pyometra-in-small-animals", "policy"),
    Rule("DC-METRITIS", ("dog", "cat"), "EMERGENCY",
         _r(r"just (whelped|given birth|kittened|had (her )?(puppies|kittens))|gave birth|whelped|kittened|after (birth|whelping|delivery|giving birth)|postpartum|post-partum|nursing|"
            r"γεννησε|μετα τον τοκετο|λοχεια|θηλαζ",
            r"foul|smelly|stinking|bad smell|pus|green|dark red|black|βρωμ|πυο",
            r"discharge|εκκριμα"),
         "Foul or pus-like discharge after giving birth can be a womb infection (metritis): go to a vet now, especially if the mother is off food, feverish or ignoring her young.",
         "Δυσάρεστης οσμής ή πυώδες έκκριμα μετά τον τοκετό μπορεί να είναι μητρίτιδα: πήγαινε αμέσως σε κτηνίατρο, ιδίως αν η μητέρα δεν τρώει, έχει πυρετό ή αγνοεί τα μικρά.",
         MSD + "reproductive-system/female-reproductive-tract-disorders-in-small-animals/metritis-in-small-animals", "policy", immediate=True),
    Rule("DC-MASTITIS", ("dog", "cat"), "URGENT",
         _r(r"mammary|breast|teat|nipple|milk gland|glands?\b|μαστ|θηλη|μαστικ",
            r"(hard|firm|hot|warm|swollen|painful|red|purple|dark|black|ulcerat\w*|bloody|pus)\b|bloody milk|milk (is )?(bloody|yellow|green|thick|clotted)|σκληρ|ζεστ|πρησμεν|κοκκιν|ματωμεν|πυο|μωβ|μαυρ"),
         "A hard, hot, painful or dark mammary gland, or bloody or pus-like milk, can be mastitis: see a vet today, and go now if the mother is sick or the gland is dark or open.",
         "Σκληρός, ζεστός, επώδυνος ή σκούρος μαστός ή ματωμένο ή πυώδες γάλα μπορεί να είναι μαστίτιδα: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν η μητέρα είναι άρρωστη ή ο μαστός σκούρος ή ανοιχτός.",
         MSD + "reproductive-system/female-reproductive-tract-disorders-in-small-animals/mastitis-in-small-animals", "policy",
         exclude=r"\bglands?\b.{0,15}(anal|saliva|thyroid)|anal gland|σαλιογον"),
    Rule("DOG-LABOUR-LONG", ("dog", "cat"), "EMERGENCY",
         _r(r"(more than|over|for|about|nearly|almost) .{0,6}(30 ?(min|minutes)|half an hour|an hour|1 hour|2 hours|two hours|3 hours|4 hours|four hours|\b([2-9]|\d\d) hours).{0,50}(push\w*|contraction\w*|strain\w*|labou?r)|"
            r"(push\w*|contraction\w*|strain\w*|labou?r).{0,50}(more than|over|for|about|nearly|almost) .{0,6}(30 ?(min|minutes)|half an hour|an hour|1 hour|2 hours|two hours|3 hours|4 hours|four hours|\b([2-9]|\d\d) hours)|"
            r"(dark|black)[- ]?green (discharge|fluid)|green (discharge|fluid) .{0,20}before|\b(7[1-9]|[89]\d) days (pregnant|gestation|since (mating|breeding))|overdue|past (her )?due date|"
            r"πρασιν\w* εκκριμα|εχει περασει η ημερομηνια|\b(7[1-9]|[89]\d) μερες εγκυμοσυνη",
            r"puppy|puppies|pup\b|kitten|pregnan|labou?r|whelp|queen|birth|contraction|delivery|gestation|mating|breeding|κουταβ|γατακ|εγκυ|γεννα|τοκετ|συσπασ|οιστρ"),
         "Hard pushing for 30 minutes with no baby, more than 2 hours of weak contractions between babies, dark green discharge before a birth, or going past about 70 days is an emergency: go to a vet now.",
         "Έντονο σπρώξιμο 30 λεπτών χωρίς μωρό, πάνω από 2 ώρες αδύναμων συσπάσεων ανάμεσα στα μωρά, σκούρο πράσινο έκκριμα πριν τη γέννα ή πέρασμα των 70 ημερών είναι επείγον: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "reproductive-system/pregnancy-and-parturition-in-small-animals/dystocia-in-small-animals", "policy", immediate=True),
    Rule("DC-PARAPHIMOSIS", ("dog", "cat"), "EMERGENCY",
         _r(r"penis|sheath|prepuce|penile|πεος|θηκη του πεους|ποσθη",
            r"stuck out|out of (the )?(sheath|prepuce)|won'?t (go|retract|slide) back|will not (go|retract) back|can'?t (retract|go back)|not retract\w*|exposed|dry and|swollen|εξω απο τη θηκη|δεν επιστρεφει|δεν μπαινει μεσα|πρησμενο|ξερο"),
         "A penis that stays out of its sheath swells and dries within hours and can be damaged permanently: do not force it back, keep your pet from licking it and go to a vet now.",
         "Πέος που μένει έξω από τη θήκη πρήζεται και στεγνώνει μέσα σε ώρες και μπορεί να πάθει μόνιμη βλάβη: μην το πιέσεις πίσω, μην αφήσεις το ζώο να το γλείφει και πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "reproductive-system/male-reproductive-disorders-in-small-animals/disorders-of-the-penis-and-prepuce-in-small-animals", "policy", immediate=True),
    Rule("DC-PRIAPISM", ("dog", "cat"), "EMERGENCY",
         _r(r"erection.{0,50}(hours|won'?t|will not|not going down|doesn'?t go down|constant\w*|persist\w*|since)|(won'?t|will not|doesn'?t) (go|stay) down.{0,20}erect|στ[ιυ]ση.{0,40}(ωρες|δεν περναει|συνεχως)"),
         "An erection that does not go down can damage the penis permanently: go to a vet now.",
         "Στύση που δεν υποχωρεί μπορεί να προκαλέσει μόνιμη βλάβη στο πέος: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "reproductive-system/male-reproductive-disorders-in-small-animals/disorders-of-the-penis-and-prepuce-in-small-animals", "policy", immediate=True),
    Rule("DC-SCROTUM", ("dog", "cat"), "EMERGENCY",
         _r(r"scrotum|testicle|testicles|testis|balls\b|όρχ|ορχ|οσχεο",
            r"swollen|swelling|red\b|painful|hot|warm|hard\b|twisted|bruis\w*|πρησμεν|πρηξιμο|κοκκιν|πονα|ζεστ|σκληρ"),
         "A swollen, red, hot or painful scrotum or testicle can be infection or a twisted testicle that is permanently damaged within hours: go to a vet now.",
         "Πρησμένο, κόκκινο, ζεστό ή επώδυνο όσχεο ή όρχις μπορεί να είναι λοίμωξη ή στρέβλωση που προκαλεί μόνιμη βλάβη μέσα σε ώρες: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "reproductive-system/male-reproductive-disorders-in-small-animals", "policy", immediate=True),
    Rule("DC-ABORTION", ("dog", "cat"), "URGENT",
         _r(r"aborted|abortion|miscarr\w*|stillborn|still-born|dead (puppies|pups|kittens|fetus|foetus)|lost (the |her |all )?(puppies|pups|kittens|litter|pregnancy)|"
            r"εκτρωση|αποβαλ|νεκρα (κουταβια|γατακια|εμβρυα)|γεννησε νεκρα"),
         "Losing a pregnancy or stillbirths in a dog or cat can be an infection such as brucellosis that also affects people, and the mother can become ill: see a vet today and wear gloves when handling fluids or tissue.",
         "Η απώλεια κύησης ή τα νεκρά νεογνά σε σκύλο ή γάτα μπορεί να οφείλονται σε λοίμωξη όπως η βρουκέλλωση που προσβάλλει και ανθρώπους και η μητέρα μπορεί να αρρωστήσει: δες κτηνίατρο σήμερα και φόρα γάντια όταν πιάνεις υγρά ή ιστούς.",
         MSD + "reproductive-system/pregnancy-and-parturition-in-small-animals", "policy"),
    Rule("DC-VULVA-MASS", ("dog", "cat"), "URGENT",
         _r(r"(vulva|vulvar|vagina|vaginal|κολπ|αιδοι).{0,40}(mass|tissue|swelling|bulg\w*|protrud\w*|prolaps\w*|sticking out|coming out|μαζα|ιστος|προβαλλ|βγαινει)|(μαζα|ιστος|κατι).{0,25}(βγαινει|προβαλλ|εξεχει).{0,25}(κολπ|αιδοι)|(mass|tissue|something|pink|red).{0,25}(sticking out|coming out|protruding|bulging).{0,25}(vulva|vagina)"),
         "Tissue bulging from the vulva can be vaginal hyperplasia, a prolapse or a tumour and can block urination: see a vet today, and go now if she cannot pee or the tissue is dark or bleeding.",
         "Ιστός που προβάλλει από την αιδοίο μπορεί να είναι υπερπλασία του κόλπου, πρόπτωση ή όγκος και μπορεί να εμποδίσει την ούρηση: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν δεν μπορεί να ουρήσει ή ο ιστός είναι σκούρος ή αιμορραγεί.",
         MSD + "reproductive-system/female-reproductive-tract-disorders-in-small-animals/vaginal-disorders-in-small-animals", "policy"),
    # ---- from the MSD veterinary Digestive, Respiratory, Circulatory and Nervous System pages (policy basis, levels inferred, vet to confirm)
    Rule("DC-ESOPH-FB", ("dog", "cat"), "EMERGENCY",
         _r(r"bone|needle|fish ?hook|hook|stick|rawhide|chew|skewer|καρφιτσα|αγκιστρι|κοκαλο|ξυλο",
            r"drool\w*|gag\w*|can'?t swallow|cannot swallow|trouble swallowing|difficulty swallowing|keeps swallowing|retch\w*|regurgitat\w*|σαλιαριζ|πνιγ|δυσκολια στην καταποση|δεν μπορει να καταπιει"),
         "Trouble swallowing or drooling after a bone, hook, needle or stick can mean it is stuck in the gullet and can perforate it: go to a vet now.",
         "Δυσκολία στην κατάποση ή σάλιο μετά από κόκαλο, αγκίστρι, βελόνα ή ξύλο μπορεί να σημαίνει ότι έχει κολλήσει στον οισοφάγο και μπορεί να τον τρυπήσει: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "digestive-system/diseases-of-the-esophagus-in-small-animals/esophageal-foreign-bodies-in-small-animals", "policy", immediate=True),
    Rule("CAT-PANLEUK", ("cat",), "EMERGENCY",
         _r(r"kitten|young cat|unvaccinated|not vaccinated|shelter|stray|γατακ|ανεμβολιαστ|αδεσποτ",
            r"vomit\w*|εμετ",
            r"not eating|won'?t eat|refus\w* (food|to eat)|fever|letharg|drool\w*|cold|δεν τρωει|πυρετ|ληθαργ|σαλιαριζ"),
         "A young or unvaccinated cat that vomits and refuses food can have feline panleukopenia (a parvo-like virus): go to a vet now and keep it away from other cats.",
         "Νεαρή ή ανεμβολίαστη γάτα που κάνει εμετό και αρνείται το φαγητό μπορεί να έχει πανλευκοπενία (ιός σαν της παρβοϊκής): πήγαινε αμέσως σε κτηνίατρο και κράτα την μακριά από άλλες γάτες.",
         MSD + "digestive-system/infectious-diseases-of-the-gastrointestinal-tract-in-small-animals/feline-panleukopenia", "policy", immediate=True),
    Rule("CAT-NOT-EATING-1D", ("cat",), "URGENT",
         _r(r"(hasn'?t|has not|didn'?t|did not|won'?t|not|stopped|refus\w*) (eaten|eating|eat)\b.{0,30}(since yesterday|all day|today|24 hours|a day|overnight|last night)|(since yesterday|all day|24 hours|a day).{0,25}(no|not|without) (food|eating|appetite)|δεν εχει φαει απο χθες|δεν τρωει απο χθες|δεν εφαγε σημερα"),
         "A cat that stops eating can develop liver failure (hepatic lipidosis) within days, especially if overweight: see a vet today and do not force-feed or give supplements at home.",
         "Γάτα που σταματά να τρώει μπορεί να πάθει ηπατική λιπίδωση μέσα σε λίγες μέρες, ιδίως αν είναι υπέρβαρη: δες κτηνίατρο σήμερα και μην την αναγκάσεις να φάει ούτε να της δώσεις συμπληρώματα στο σπίτι.",
         MSD + "digestive-system/hepatic-disease-in-small-animals/feline-hepatic-lipidosis", "policy"),
    Rule("DC-PERITONITIS", ("dog", "cat"), "EMERGENCY",
         _r(r"belly|abdomen|stomach|tummy|κοιλια",
            r"swollen|tense|hard|bloated|distended|painful|φουσκωμεν|σκληρ|τεντωμεν|πονα",
            r"(surgery|operation|stitches|spay\w*|neuter\w*|bite|bitten|wound|puncture|fever|collaps\w*|χειρουργει|ραμματα|δαγκωμα|τραυμα|πυρετ|καταρρευ)"),
         "A swollen, painful belly with fever, a recent operation, a bite or collapse can be peritonitis, which gets worse within hours: go to a vet now.",
         "Φουσκωμένη, επώδυνη κοιλιά με πυρετό, πρόσφατη επέμβαση, δάγκωμα ή κατάρρευση μπορεί να είναι περιτονίτιδα, που χειροτερεύει μέσα σε ώρες: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "digestive-system/peritonitis/peritonitis-in-animals", "policy", immediate=True),
    Rule("DC-MELENA", ("dog", "cat"), "URGENT",
         _r(r"black (tarry |sticky |tar-like )?(stool|poop|faeces|feces|droppings)|tarry (stool|poop)|melena|μαυρα κοπρανα|μαυρη κακα|πισσα"),
         "Black, tarry stool means digested blood from higher up in the gut (ulcer, poison, bleeding): see a vet today, and go now if the gums are pale or the pet is weak.",
         "Μαύρα πισσώδη κόπρανα σημαίνουν χωνεμένο αίμα από το πάνω μέρος του εντέρου (έλκος, δηλητηρίαση, αιμορραγία): δες κτηνίατρο σήμερα και πήγαινε αμέσως αν τα ούλα είναι χλωμά ή το ζώο αδύναμο.",
         MSD + "digestive-system/diseases-of-the-stomach-in-small-animals/gastrointestinal-ulcers-in-small-animals", "policy"),
    Rule("DC-ANAL-SAC", ("dog", "cat"), "URGENT",
         _r(r"anus|anal|rear end|bottom|butt|πρωκτ|ουρα|πισω",
            r"swollen|swelling|red\b|purple|draining|burst|ruptur\w*|hole|pus|bleeding|painful|πρησμεν|κοκκιν|μωβ|πυο|τρυπα|αιμορραγ|πονα"),
         "A swollen, red, purple, draining or painful area beside the anus can be an anal-sac abscess: see a vet today.",
         "Πρησμένη, κόκκινη, μωβ, επώδυνη ή με έκκριμα περιοχή δίπλα στον πρωκτό μπορεί να είναι απόστημα πρωκτικού αδένα: δες κτηνίατρο σήμερα.",
         MSD + "digestive-system/diseases-of-the-rectum-and-anus/anal-sac-disease-in-dogs-and-cats", "policy",
         exclude=r"vomit|bite on|insect"),
    Rule("DC-REGURG-COUGH", ("dog", "cat"), "EMERGENCY",
         _r(r"regurgitat\w*|brings? (up|back) (food|water)|food comes (back )?up|undigested food|βγαζει (το )?φαγητο|ξερναει αχωνευτο",
            r"cough\w*|fast breath\w*|breathing fast|rapid breath\w*|fever|nasal discharge|βηχ|γρηγορη αναπνοη|πυρετ"),
         "Bringing food back up together with coughing, fast breathing or fever can mean food has gone into the lungs (aspiration pneumonia): go to a vet now.",
         "Επανέρχεται φαγητό μαζί με βήχα, γρήγορη αναπνοή ή πυρετό: μπορεί το φαγητό να έχει μπει στους πνεύμονες (πνευμονία εισρόφησης): πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "digestive-system/diseases-of-the-esophagus-in-small-animals/dilatation-of-the-esophagus-in-small-animals", "policy", immediate=True),
    Rule("DC-ASPIRATION", ("dog", "cat"), "URGENT",
         _r(r"(inhaled|breathed in|aspirat\w*|went down the wrong way|wrong pipe|choked on|gagged on).{0,40}(vomit|medicine|medication|liquid|water|milk|food|oil|syringe)|(syringe|squirted|forced).{0,40}(too fast|quickly).{0,30}(medicine|liquid|water)|εισροφ|ειρροφ|πηγε στραβα|πνιγηκε με",),
         "Breathing in liquid, vomit or food can cause pneumonia hours to days later even if the pet looks fine now: see a vet today and go now if there is coughing, fast or hard breathing or blue gums.",
         "Η εισρόφηση υγρού, εμέτου ή φαγητού μπορεί να προκαλέσει πνευμονία ώρες ή μέρες αργότερα ακόμη κι αν το ζώο φαίνεται καλά τώρα: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν υπάρχει βήχας, γρήγορη ή δύσκολη αναπνοή ή μπλε ούλα.",
         MSD + "respiratory-system/respiratory-diseases-of-small-animals/aspiration-pneumonia", "policy"),
    Rule("DC-NO-LIE-DOWN", ("dog", "cat"), "EMERGENCY",
         _r(r"won'?t lie down|will not lie down|refuses to lie down|can'?t lie down|stays? (standing|upright|sitting up)|sits? up all night|sleeping (sitting|upright|with (her|his) (head|neck) (up|stretched))|neck stretched out|elbows (out|stuck out)|δεν ξαπλωνει|δεν θελει να ξαπλωσει|κοιμαται καθιστος",
            r"breath\w*|pant\w*|gasp\w*|αναπνο|λαχανιαζ"),
         "A pet that will not lie down and breathes hard or fast is struggling for air (fluid in the lungs or chest): go to a vet now and keep it calm.",
         "Ζώο που δεν ξαπλώνει και αναπνέει δύσκολα ή γρήγορα παλεύει για αέρα (υγρό στους πνεύμονες ή στον θώρακα): πήγαινε αμέσως σε κτηνίατρο και κράτα το ήρεμο.",
         MSD + "respiratory-system/respiratory-diseases-of-small-animals/pulmonary-edema-in-small-animals", "policy", immediate=True),
    Rule("DOG-STRIDOR", ("dog", "cat"), "EMERGENCY",
         _r(r"stridor|wheez\w*|harsh (noisy )?breath\w*|noisy breath\w*|raspy breath\w*|voice (changed|is hoarse)|bark (has )?changed|hoarse bark|σφυριγμα|βραχνη αναπνοη|αλλαξε το γαβγισμα",
            r"collaps\w*|blue|purple|struggl\w*|distress\w*|can'?t breathe|hard to breathe|gasping|heat|hot day|καταρρε|κατερρε|μπλε|μωβ|δυσκολευεται|δεν μπορει να αναπνευσει"),
         "Noisy, harsh breathing with distress, blue gums or collapse can mean a blocked airway (for example laryngeal paralysis): go to a vet now and keep the pet cool and calm.",
         "Θορυβώδης, τραχιά αναπνοή με δυσφορία, μπλε ούλα ή κατάρρευση μπορεί να σημαίνει απόφραξη αεραγωγού (π.χ. παράλυση λάρυγγα): πήγαινε αμέσως σε κτηνίατρο και κράτα το δροσερό και ήρεμο.",
         MSD + "respiratory-system/respiratory-diseases-of-small-animals/laryngeal-paralysis-in-small-animals", "policy", immediate=True),
    Rule("DC-NOSE-ONESIDE", ("dog", "cat"), "URGENT",
         _r(r"(one|single) nostril|from one side of (the|his|her) nose|pawing (at )?(the |his |her )?(nose|face)|sudden sneez\w*|ενα ρουθουνι|ξυνει τη μυτη",
            r"discharge|bleed\w*|blood|mucus|pus|sneez\w*|εκκριμα|αιμα|αιμορραγ|φταρνισμ"),
         "Discharge from one nostril, especially with sudden sneezing or pawing at the nose, can be a foreign body: see a vet today.",
         "Έκκριμα από το ένα ρουθούνι, ιδίως με ξαφνικό φτάρνισμα ή ξύσιμο της μύτης, μπορεί να είναι ξένο σώμα: δες κτηνίατρο σήμερα.",
         MSD + "respiratory-system/respiratory-diseases-of-small-animals/rhinitis-and-sinusitis-in-small-animals", "policy"),
    Rule("DC-KITTEN-URI", ("cat",), "URGENT",
         _r(r"kitten|old cat|senior cat|γατακ|ηλικιωμεν",
            r"sneez\w*|nasal discharge|runny (nose|eyes?)|eye discharge|φταρνισμ|μυξα|δακρυζ",
            r"not eating|won'?t eat|no appetite|squint\w*|eye (is )?(closed|cloudy)|dehydrat\w*|δεν τρωει|μισοκλειστο|θολο|αφυδατ"),
         "A kitten or old cat with a cold that stops eating, squints or is dehydrated needs a vet today: they can go downhill quickly and eye ulcers need treatment.",
         "Γατάκι ή ηλικιωμένη γάτα με συνάχι που σταματά να τρώει, μισοκλείνει το μάτι ή αφυδατώνεται χρειάζεται κτηνίατρο σήμερα: μπορεί να χειροτερέψει γρήγορα και τα έλκη στα μάτια θέλουν θεραπεία.",
         MSD + "respiratory-system/respiratory-diseases-of-small-animals/feline-upper-respiratory-disease-complex", "policy"),
    Rule("CAT-SADDLE", ("cat",), "EMERGENCY",
         _r(r"hind (legs?|limbs?)|back (legs?|limbs?)|rear (legs?|limbs?)|πισω ποδια|πισω ποδαρια",
            r"sudden\w*|suddenly|all of a sudden|dragging|paralys\w*|can'?t use|won'?t use|cold|pale|blue|purple|screaming|crying|howl\w*|ξαφνικα|παραλυ|κρυα|χλωμα|μπλε|μωβ|ουρλιαζ|κλαιει"),
         "A cat that suddenly cannot use its back legs, especially if they are cold, pale or painful, may have a blood clot (often from heart disease): go to a vet now and do not give any human painkiller or aspirin.",
         "Γάτα που ξαφνικά δεν μπορεί να χρησιμοποιήσει τα πίσω πόδια, ιδίως αν είναι κρύα, χλωμά ή επώδυνα, μπορεί να έχει θρόμβο (συχνά από καρδιοπάθεια): πήγαινε αμέσως σε κτηνίατρο και μην δώσεις ανθρώπινο παυσίπονο ή ασπιρίνη.",
         MSD + "circulatory-system/cardiovascular-diseases-and-disorders-of-small-animals/arterial-thromboembolism-in-small-animals", "policy", immediate=True),
    Rule("DC-PETECHIAE", ("dog", "cat"), "URGENT",
         _r(r"petechia\w*|pinpoint (red|purple)|tiny (red|purple) (spots|dots)|red (spots|dots) on (the )?(gums|belly|skin)|bruis\w*|bleeding gums|blood spots|κοκκινες (τελιες|βουλιτσες)|μωλωπ|αιμορραγουν τα ουλα"),
         "Pinpoint red or purple spots, bruises or bleeding gums can mean the blood is not clotting (low platelets, poison): see a vet today, and go now if there is black stool, blood in urine, bleeding that will not stop or weakness.",
         "Κόκκινες ή μωβ τελίες, μώλωπες ή αιμορραγία στα ούλα μπορεί να σημαίνουν ότι το αίμα δεν πήζει (χαμηλά αιμοπετάλια, δηλητηρίαση): δες κτηνίατρο σήμερα και πήγαινε αμέσως αν υπάρχουν μαύρα κόπρανα, αίμα στα ούρα, αιμορραγία που δεν σταματά ή αδυναμία.",
         MSD + "circulatory-system/hemostatic-disorders-in-small-animals/thrombocytopenia-in-small-animals", "policy"),
    Rule("DC-SUDDEN-BLIND", ("dog", "cat"), "URGENT",
         _r(r"sudden\w*|overnight|all of a sudden|ξαφνικα",
            r"blind\w*|can'?t see|cannot see|bumping into|walking into (things|walls)|τυφλ|δεν βλεπει|χτυπαει πανω"),
         "Sudden blindness can come from high blood pressure, bleeding in the eye or a brain problem and is time-sensitive: see a vet today.",
         "Η ξαφνική τύφλωση μπορεί να οφείλεται σε υψηλή πίεση, αιμορραγία στο μάτι ή πρόβλημα στον εγκέφαλο και είναι επείγουσα: δες κτηνίατρο σήμερα.",
         MSD + "circulatory-system/hypertension/hypertension-in-small-animals", "policy"),
    Rule("DC-NEURO-FEVER", ("dog", "cat"), "EMERGENCY",
         _r(r"neck (is )?(stiff|painful|sore|rigid)|stiff neck|cries? when (the )?(neck|head)|head (arched|tilted back)|head tilt|λαιμος.{0,25}(σκληρ|πονα)|γερμενο κεφαλι",
            r"fever|letharg\w*|not eating|won'?t eat|no appetite|confus\w*|seizure\w*|drowsy|unresponsive|πυρετ|ληθαργ|δεν τρωει|συγχυσ|σπασμ|νυσταλ"),
         "Neck pain or a head tilt with fever, lethargy, not eating or confusion can mean infection or inflammation of the brain or its lining: go to a vet now.",
         "Πόνος στον αυχένα ή γερμένο κεφάλι με πυρετό, λήθαργο, άρνηση φαγητού ή σύγχυση μπορεί να σημαίνει λοίμωξη ή φλεγμονή του εγκεφάλου ή των μηνίγγων: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "nervous-system/meningitis-myelitis-and-encephalitis-in-small-animals", "policy", immediate=True),
    Rule("DC-TICK-PARALYSIS", ("dog", "cat"), "EMERGENCY",
         _r(r"tick|τσιμπουρ",
            r"weak\w*|wobbl\w*|can'?t stand|cannot stand|hind (legs?|limbs?)|voice (change|is different)|bark (change|is different)|gagging|difficulty breathing|παραλυ|αδυναμ|τρεμουλ|δεν στεκεται|πισω ποδια|αλλαγη φωνης"),
         "A tick with new weakness, wobbly back legs, a changed voice or breathing trouble can be tick paralysis, which can stop breathing within a day or two: go to a vet now and remove any ticks you can see safely.",
         "Τσιμπούρι με νέα αδυναμία, ασταθή πίσω πόδια, αλλαγή στη φωνή ή δυσκολία στην αναπνοή μπορεί να είναι παράλυση από τσιμπούρι, που μπορεί να σταματήσει την αναπνοή μέσα σε μία με δύο μέρες: πήγαινε αμέσως σε κτηνίατρο και αφαίρεσε τσιμπούρια που βλέπεις με ασφάλεια.",
         MSD + "nervous-system/tick-paralysis", "policy", immediate=True),
    Rule("DC-HYPOGLY-EPISODE", ("dog", "cat", "other"), "URGENT",
         _r(r"weak\w*|collaps\w*|wobbl\w*|staring|glazed|drool\w*|pawing (at )?(the |his |her )?mouth|trembl\w*|twitch\w*|αδυναμ|καταρρε|κατερρε|σαλιαριζ|τρεμ",
            r"(better|recover\w*|fine|normal) (after|when|once) (eating|he eats|she eats|food)|improv\w* (after|with) (eating|food)|episodes?|comes? and goes|on and off|πιο καλα μετα το φαγητο|επεισοδι"),
         "Episodes of weakness, staring or collapse that ease after eating can be dangerously low blood sugar (for example an insulin-producing tumour): see a vet today, and go now if there is a seizure or the pet is unresponsive.",
         "Επεισόδια αδυναμίας, βλέμματος στο κενό ή κατάρρευσης που περνούν μετά το φαγητό μπορεί να είναι επικίνδυνα χαμηλό σάκχαρο (π.χ. όγκος που παράγει ινσουλίνη): δες κτηνίατρο σήμερα και πήγαινε αμέσως αν υπάρχει σπασμός ή το ζώο δεν ανταποκρίνεται.",
         MSD + "endocrine-system/the-pancreas/insulinoma-in-dogs-and-cats", "policy"),
    Rule("DC-DISTEMPER", ("dog",), "EMERGENCY",
         _r(r"unvaccinated|not vaccinated|no vaccin\w*|puppy|stray|ανεμβολιαστ|κουταβ|αδεσποτ",
            r"chewing[- ]?gum|jaw (chewing|champing)|twitch\w*|muscle (jerk|twitch)\w*|seizure\w*|circling|head tilt|σπασμ|συσπασ|κυκλ",
            r"fever|nasal discharge|eye discharge|diarrh\w*|cough\w*|πυρετ|εκκριμα|διαρρο|βηχ"),
         "An unvaccinated dog or puppy with fever, discharge and twitching or fits may have distemper: go to a vet now (call ahead) and keep it away from other dogs.",
         "Ανεμβολίαστος σκύλος ή κουτάβι με πυρετό, εκκρίματα και συσπάσεις ή σπασμούς μπορεί να έχει νόσημα του σκύλου (distemper): πήγαινε αμέσως σε κτηνίατρο (τηλεφώνησε πριν) και κράτα το μακριά από άλλους σκύλους.",
         MSD + "generalized-conditions/canine-distemper", "policy", immediate=True),
    Rule("DC-TETANUS", ("dog", "cat"), "EMERGENCY",
         _r(r"stiff\w*|rigid\w*|spasm\w*|lips (drawn|pulled) back|can'?t open (his |her |the )?mouth|jaw (locked|stiff)|σκληρο|σπασμ|δεν ανοιγει το στομα",
            r"wound|puncture|bite|cut|nail|thorn|splinter|τραυμα|δαγκωμα|κοψιμο|καρφι|αγκαθι"),
         "Stiff muscles, spasms or a locked jaw after a deep wound can be tetanus: go to a vet now, early treatment matters.",
         "Σκληροί μύες, σπασμοί ή κλειδωμένο σαγόνι μετά από βαθύ τραύμα μπορεί να είναι τέτανος: πήγαινε αμέσως σε κτηνίατρο, η πρώιμη θεραπεία έχει σημασία.",
         MSD + "nervous-system/tetanus-in-animals", "policy", immediate=True),
    Rule("DC-SPINE-PAIN-NSAID", ("dog", "cat"), "URGENT",
         _r(r"back (pain|is painful|hurts)|spine|disc|slipped disc|yelp\w*|cries? when (picked up|touched|lifted)|hunched|reluctant to (jump|move|climb)|πονα\w* η μεση|πονος στη μεση|πονα\w* η ραχη|σπονδυλ|δισκο|κυφωση|φωναζει οταν",
            r"(ibuprofen|paracetamol|acetaminophen|aspirin|nurofen|panadol|depon|voltaren|naproxen|painkiller|pain killer|human pain|παυσιπονο|ιβουπροφαιν|παρακεταμολ|ασπιριν)"),
         "Do not give human painkillers for back or neck pain: they can cause bleeding ulcers and kidney damage, and the pet may hurt itself more once it feels better. Keep it strictly rested and see a vet today.",
         "Μη δώσεις ανθρώπινα παυσίπονα για πόνο στην πλάτη ή τον αυχένα: μπορεί να προκαλέσουν έλκη με αιμορραγία και νεφρική βλάβη και το ζώο μπορεί να τραυματιστεί περισσότερο όταν νιώσει καλύτερα. Κράτα το σε αυστηρή ανάπαυση και δες κτηνίατρο σήμερα.",
         MSD + "musculoskeletal-system/degenerative-disorders-of-the-spine/intervertebral-disc-disease-in-small-animals", "policy"),
    Rule("FERRET-URINE-STRAIN", ("other",), "EMERGENCY",
         _r(r"ferret|φερετ",
            r"strain\w*|squat\w*|can'?t (pee|urinate)|no urine|dribbl\w*|screaming|σφιγγεται|δεν ουρει|σταγονες"),
         "A ferret that strains to pee or passes little or no urine may have a blocked urethra: go to a vet now.",
         "Κουνάβι (ferret) που σφίγγεται για να ουρήσει ή βγάζει λίγα ή καθόλου ούρα μπορεί να έχει απόφραξη ουρήθρας: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "all-other-pets/ferrets/other-noninfectious-diseases-of-ferrets", "policy", immediate=True),
    Rule("BIRD-FLUFFED-FLOOR", ("bird",), "EMERGENCY",
         _r(r"fluffed( up)?|puffed( up)?|ruffled|on the (cage )?floor|sitting on the floor|can'?t perch|falling off (the )?perch|sleepy all day|βουλιαγμενο|φουσκωμενα φτερα|στο πατωμα|δεν στεκεται στο κοτσι"),
         "A bird that sits fluffed up, stays on the cage floor or cannot perch is seriously ill (birds hide illness until late): go to a vet who sees birds now and keep it warm and quiet.",
         "Πουλί που κάθεται φουσκωμένο, μένει στο πάτωμα του κλουβιού ή δεν μπορεί να σταθεί στο κοτσί είναι σοβαρά άρρωστο (τα πουλιά κρύβουν την ασθένεια μέχρι αργά): πήγαινε αμέσως σε κτηνίατρο που βλέπει πουλιά και κράτα το ζεστό και ήσυχο.",
         MSD + "bird-owners/disorders-and-diseases-of-birds/digestive-disorders-of-pet-birds", "policy", immediate=True),
    Rule("DC-BLOODPARASITE", ("dog", "cat"), "EMERGENCY",
         _r(r"tick|flea|mosquito|τσιμπουρ|ψυλλ|κουνουπ",
            r"dark (brown |red )?(urine|pee)|coffee[- ]colou?red (urine|pee)|red urine|jaundice|yellow (gums|eyes|skin)|pale gums|white gums|σκουρα ουρα|κοκκινα ουρα|ικτερ|κιτρινα ουλα|χλωμα ουλα",
            r"weak\w*|letharg\w*|collaps\w*|not eating|fever|breathing|αδυναμ|ληθαργ|καταρρε|κατερρε|δεν τρωει|πυρετ|αναπνο"),
         "A weak or feverish pet with pale or yellow gums or dark urine after tick, flea or mosquito exposure can have a blood parasite (babesiosis, hemoplasma, heartworm) that destroys red blood cells: go to a vet now.",
         "Αδύναμο ή πυρετικό ζώο με χλωμά ή κίτρινα ούλα ή σκούρα ούρα μετά από τσιμπούρι, ψύλλο ή κουνούπι μπορεί να έχει παράσιτο του αίματος (babesia, αιμοπλάσμωση, διροφιλάρια) που καταστρέφει τα ερυθρά: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "circulatory-system/blood-parasites/babesiosis-in-small-animals", "policy", immediate=True),
    Rule("DC-HEARTWORM-SIGNS", ("dog", "cat", "other"), "URGENT",
         _r(r"heartworm|dirofilari\w*|διροφιλαρ|καρδιακο σκουληκι|καρδιακα σκουληκια",
            r"cough\w*|tir\w* (easily|quickly)|collaps\w*|faint\w*|breath\w*|weight loss|swollen belly|βηχ|κουραζ|καταρρε|κατερρε|αναπνο|αδυνατ|φουσκωμενη κοιλια"),
         "Heartworm (spread by mosquitoes, also in cats and ferrets) with coughing, tiring easily, fainting or breathing trouble needs a vet today, and go now if there is collapse, blue or pale gums, or very dark urine.",
         "Η διροφιλαρίωση (μεταδίδεται από κουνούπια, και σε γάτες και ferrets) με βήχα, εύκολη κόπωση, λιποθυμία ή δύσκολη αναπνοή θέλει κτηνίατρο σήμερα και πήγαινε αμέσως αν υπάρχει κατάρρευση, μπλε ή χλωμά ούλα ή πολύ σκούρα ούρα.",
         MSD + "circulatory-system/heartworm-disease/heartworm-disease-in-dogs", "policy"),
    Rule("REP-STARGAZE", ("reptile",), "URGENT",
         _r(r"star ?gaz\w*|looking up|neck (twisted|bent|arched)|head (twisted|tilted)|tremor\w*|twitch\w*|seizure\w*|can'?t (walk|move) (properly|well)|κοιτα ψηλα|στριμμενος λαιμος|τρεμουλ|σπασμ"),
         "Stargazing, a twisted neck, tremors or fits in a reptile can come from poor calcium or UVB, heat, injury or infection: see a reptile vet today, and go now if there are fits or it was overheated or hurt.",
         "Το να κοιτάζει προς τα πάνω, στριμμένος λαιμός, τρεμούλιασμα ή σπασμοί σε ερπετό μπορεί να οφείλονται σε έλλειψη ασβεστίου ή UVB, θερμότητα, τραυματισμό ή λοίμωξη: δες κτηνίατρο ερπετών σήμερα και πήγαινε αμέσως αν υπάρχουν σπασμοί ή υπερθέρμανση ή τραυματισμός.",
         MSD + "all-other-pets/reptiles/disorders-and-diseases-of-reptiles", "policy"),
    # ---- from the MSD veterinary Eye and Ear pages (policy basis, levels inferred except proptosis, vet to confirm)
    Rule("DC-EYE-PROPTOSIS", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"eye.{0,25}(popped|pop\w* out|out of (its|the|his|her) socket|bulg\w* out|hanging out|came out|prolaps\w*|proptos\w*|sticking out)|(popped|prolaps\w*) .{0,10}eye|μάτι.{0,20}(βγηκε|εξω απο την κογχη|κρεμεται)|ματι.{0,20}(βγηκε|εξω απο την κογχη|κρεμεται)"),
         "An eye that has popped out of its socket must be put back by a vet as soon as possible: go now, keep the eye moist with a clean damp cloth and do not push on it.",
         "Μάτι που βγήκε από την κόγχη πρέπει να επανατοποθετηθεί από κτηνίατρο όσο το δυνατόν γρηγορότερα: πήγαινε αμέσως, κράτα το μάτι υγρό με καθαρό βρεγμένο πανί και μην το πιέζεις.",
         MSD + "eye-diseases-and-disorders/ophthalmology/prolapse-of-the-eye-in-animals", "msd", immediate=True),
    Rule("DC-EYE-GLAUCOMA", ("dog", "cat"), "EMERGENCY",
         _r(r"eye|ματι",
            r"red\b|bloodshot|cloudy|bluish|hazy|bulging|enlarg\w*|κοκκιν|θολο|μπλε|πρησμεν|μεγαλ",
            r"painful|pain\b|squint\w*|rubbing|pawing|hiding|won'?t open|big pupil|dilated pupil|large pupil|πονα|μισοκλειστ|τριβει|μεγαλη κορη|διεσταλμεν\w* κορη"),
         "A red or cloudy painful eye, especially with a big pupil, can be acute glaucoma, which can cause permanent blindness within hours: go to a vet now.",
         "Κόκκινο ή θολό επώδυνο μάτι, ιδίως με μεγάλη κόρη, μπορεί να είναι οξύ γλαύκωμα που προκαλεί μόνιμη τύφλωση μέσα σε ώρες: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "eye-diseases-and-disorders/ophthalmology/glaucoma-in-animals", "policy", immediate=True,
         exclude=r"no (pain|redness)|not (painful|red)|pink eye only"),
    Rule("DC-EYEWORM", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"(small |tiny |thin )?(white|whitish|clear) (worms?|threads?).{0,25}(eye|ματι)|worms? (in|on|moving in) (the |his |her )?eye|σκουληκ\w*.{0,20}ματι|ματι.{0,20}σκουληκ"),
         "Small worms moving on the eye surface (eyeworm, common in Europe) damage the cornea: see a vet today to have them removed.",
         "Μικρά σκουλήκια που κινούνται στην επιφάνεια του ματιού (θελαζίωση, συχνή στην Ευρώπη) βλάπτουν τον κερατοειδή: δες κτηνίατρο σήμερα για να αφαιρεθούν.",
         MSD + "eye-diseases-and-disorders/eyeworm-disease/eyeworm-disease-in-small-animals", "policy"),
    Rule("DC-EAR-HEMATOMA", ("dog", "cat", "rabbit"), "URGENT",
         _r(r"ear|flap|pinna|αυτι",
            r"swollen|swelling|balloon\w*|puffy|fluid[- ]filled|blood blister|like a pillow|πρησμεν|φουσκωσε|φουσκα|μαξιλαρι",
            r"shak\w* (his |her )?head|scratch\w*|infection|smell|τιναζ|ξυνει|μολυνση"),
         "A swollen, balloon-like ear flap after head shaking or scratching is usually an ear hematoma on top of an ear infection or allergy: see a vet within a day or two.",
         "Πρησμένο, φουσκωμένο πτερύγιο αυτιού μετά από τίναγμα του κεφαλιού ή ξύσιμο είναι συνήθως αιμάτωμα αυτιού πάνω σε ωτίτιδα ή αλλεργία: δες κτηνίατρο μέσα σε μία με δύο μέρες.",
         MSD + "ear-disorders/diseases-of-the-pinna/auricular-hematomas-in-animals", "policy"),
    Rule("DC-EAR-BRAIN", ("dog", "cat"), "EMERGENCY",
         _r(r"ear|otitis|αυτι|ωτιτιδ",
            r"head tilt|tilting (his |her )?head|circling|falling over|losing balance|loses balance|eyes? (flick|dart|jerk)\w*|γερνει το κεφαλι|κυκλους|χανει την ισορροπια|πεφτει",
            r"fever|letharg\w*|not eating|won'?t eat|no appetite|dull|drowsy|seizure\w*|πυρετ|ληθαργ|δεν τρωει|νυσταλ|σπασμ"),
         "Ear signs with loss of balance or a head tilt plus fever, lethargy or not eating can mean the infection is reaching the inner ear and brain: go to a vet now.",
         "Συμπτώματα αυτιού με απώλεια ισορροπίας ή γερμένο κεφάλι μαζί με πυρετό, λήθαργο ή άρνηση φαγητού μπορεί να σημαίνουν ότι η λοίμωξη φτάνει στο έσω αυτί και στον εγκέφαλο: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "ear-disorders/otitis-media-and-interna/otitis-media-and-interna-in-animals", "msd", immediate=True),
    Rule("DC-EAR-FOREIGN", ("dog", "cat", "rabbit"), "URGENT",
         _r(r"ear|αυτι",
            r"grass (seed|awn)|foxtail|seed|stuck|something in|foreign|water got|σπορος|ακανθα|αγκαθι|ξυλαρι|κολλησε|κατι μεσα|ξενο σωμα",
            r"shak\w*|scratch\w*|cry\w*|yelp\w*|pain\w*|tilt\w*|τιναζ|ξυνει|πονα|κλαιει|γερν"),
         "Something stuck in the ear canal (a grass seed, for example) is painful and can burst the eardrum: see a vet today and do not push anything into the ear.",
         "Κάτι κολλημένο στον ακουστικό πόρο (π.χ. σπόρος χόρτου) πονάει και μπορεί να σπάσει το τύμπανο: δες κτηνίατρο σήμερα και μην βάζεις τίποτα μέσα στο αυτί.",
         MSD + "ear-disorders/otitis-externa/otitis-externa-in-animals", "policy"),
    # ---- from the MSD veterinary Endocrine and Metabolic pages (policy basis, vet to confirm)
    Rule("DC-DKA", ("dog", "cat"), "EMERGENCY",
         _r(r"diabet\w*|insulin|sugar disease|drinking (much |a lot )?more.{0,40}(pee|urinat)|διαβητ|ινσουλιν",
            r"vomit\w*|not eating|won'?t eat|off (his|her) food|refus\w* (food|to eat)|weak\w*|dehydrat\w*|collaps\w*|εμετ|δεν τρωει|αδυναμ|αφυδατ|καταρρευ"),
         "A diabetic pet (or one drinking and peeing far more) that now vomits, refuses food or is weak can be in diabetic ketoacidosis, which needs hospital drips and insulin: go to a vet now and do not give insulin on your own.",
         "Διαβητικό ζώο (ή ζώο που πίνει και ουρεί πολύ περισσότερο) που τώρα κάνει εμετό, αρνείται το φαγητό ή είναι αδύναμο μπορεί να έχει διαβητική κετοοξέωση που χρειάζεται ορούς και ινσουλίνη στο νοσοκομείο: πήγαινε αμέσως σε κτηνίατρο και μη δώσεις ινσουλίνη μόνος σου.",
         MSD + "endocrine-system/the-pancreas/diabetes-mellitus-in-dogs-and-cats", "msd", immediate=True),
    Rule("DC-INSULIN-LOW", ("dog", "cat"), "EMERGENCY",
         _r(r"insulin|ινσουλιν",
            r"double dose|too much|extra dose|overdos\w*|didn'?t eat|not eating|skipped (a )?meal|vomited|wobbl\w*|twitch\w*|trembl\w*|seizure\w*|collaps\w*|unresponsive|διπλη δοση|παραπανω|δεν εφαγε|εμετ|τρεμ|σπασμ|καταρρευ"),
         "Too much insulin, or insulin given without a meal, can drop blood sugar dangerously: if the pet is wobbly, twitching, very sleepy or fitting go to a vet now.",
         "Υπερβολική ινσουλίνη ή ινσουλίνη χωρίς γεύμα μπορεί να ρίξει επικίνδυνα το σάκχαρο: αν το ζώο τρικλίζει, τρέμει, νυστάζει πολύ ή έχει σπασμούς πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "endocrine-system/the-pancreas/diabetes-mellitus-in-dogs-and-cats", "policy", immediate=True),
    Rule("DOG-ADDISON-KNOWN", ("dog",), "EMERGENCY",
         _r(r"addison\w*|hypoadrenocortic\w*|αντισον|υποαδρεν",
            r"vomit\w*|diarrh\w*|weak\w*|collaps\w*|letharg\w*|shak\w*|missed|not eating|εμετ|διαρρο|αδυναμ|καταρρε|κατερρε|ληθαργ|δεν τρωει"),
         "A dog with Addison's disease that is vomiting, weak or has missed its injection or tablets can slip into an Addisonian crisis within hours: go to a vet now.",
         "Σκύλος με νόσο του Addison που κάνει εμετό, είναι αδύναμος ή έχει χάσει την ένεση ή τα χάπια του μπορεί να μπει σε κρίση Addison μέσα σε ώρες: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "endocrine-system/the-adrenal-glands/addison-disease-hypoadrenocorticism-in-animals", "msd", immediate=True),
    Rule("DC-ENEMA-PHOSPHATE", ("dog", "cat"), "EMERGENCY",
         _r(r"fleet|phosphate enema|enema|piperazine|κλισμα|υποκλυσμ",
            r"gave|given|used|put|dose|tried|εδωσα|εβαλα|χρησιμοποιησα"),
         "Human enemas (especially phosphate ones such as Fleet) can drop calcium and upset salts dangerously in cats and small dogs: go to a vet now.",
         "Τα ανθρώπινα υποκλυσμάτα (ιδίως με φωσφορικά) μπορούν να ρίξουν επικίνδυνα το ασβέστιο και να διαταράξουν τα άλατα σε γάτες και μικρούς σκύλους: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "metabolic-disorders/disorders-of-calcium-metabolism/hypocalcemia-in-small-animals", "policy", immediate=True),
    Rule("DC-HYPOCA-SIGNS", ("dog", "cat"), "URGENT",
         _r(r"rubb\w* (his |her )?(face|muzzle)|twitch\w* (ears?|face)|(ears?|face) twitch\w*|muscle spasm\w*|stiff (walk|gait|legs)|tetany|τρεμουλο (στο )?προσωπο|σπασμοι (στους )?μυες|σκληρο βαδισμα"),
         "Face rubbing, twitching of the ears or face, or a stiff walk can be low blood calcium (after surgery, in kidney or pancreas disease, in nursing mothers): see a vet today, and go now if the muscles lock or there is a fit.",
         "Τρίψιμο του προσώπου, σύσπαση αυτιών ή προσώπου ή σκληρό βάδισμα μπορεί να είναι χαμηλό ασβέστιο (μετά από χειρουργείο, σε νεφρική νόσο ή παγκρεατίτιδα, σε θηλάζουσες μητέρες): δες κτηνίατρο σήμερα και πήγαινε αμέσως αν μπλοκάρουν οι μύες ή υπάρξει σπασμός.",
         MSD + "endocrine-system/the-parathyroid-glands-and-disorders-of-calcium-regulation-in-dogs-and-cats/hypocalcemia-in-dogs-and-cats", "policy"),
    # ---- from the MSD first-aid, emergency, toxicology and infection pages (policy basis unless msd, vet to confirm)
    Rule("DC-MUSHROOM", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"mushroom|toadstool|fung(us|i)\b|μανιταρ",
            r"ate|eaten|eating|chew\w*|swallow\w*|nibbl\w*|licked|εφαγε|τρωει|μασουλ|καταπι|γλειψε"),
         "Eating a wild mushroom is an emergency even if the pet looks well: the deadly kinds cause no signs for 6 to 12 hours and then liver failure. Go to a vet now, take a photo or a sample of the mushroom and note the time.",
         "Η κατανάλωση άγριου μανιταριού είναι επείγουσα ακόμη κι αν το ζώο φαίνεται καλά: τα θανατηφόρα είδη δεν δίνουν συμπτώματα για 6 έως 12 ώρες και μετά προκαλούν ηπατική ανεπάρκεια. Πήγαινε αμέσως σε κτηνίατρο, πάρε φωτογραφία ή δείγμα του μανιταριού και σημείωσε την ώρα.",
         MSD + "toxicology/poisonous-mushrooms/overview-of-mushrooms-toxic-to-animals", "msd", immediate=True,
         exclude=r"(supermarket|shop[- ]bought|store[- ]bought|cooked|button|champignon|pizza|soup|stir|sauce|με μανιταρια ψητ|πιτσα|σουπα)"),
    Rule("ALL-DROWNING", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"drown\w*|nearly drowned|fell (in|into) (the )?(pool|sea|water|lake|river|well)|(pool|sea|lake|river|well|sewage|water).{0,25}(under ?water|sank|submerged|went under|pulled (him|her|it) out)|submerged|πνιγ\w*.{0,15}(νερο|θαλασσα|πισινα)|επεσε στην (πισινα|θαλασσα|λιμνη|γεφυρα)|βυθιστηκε"),
         "Even a short spell under water can cause lung damage that shows hours later: go to a vet now, even if the pet looks fine and is breathing normally.",
         "Ακόμη και μικρό διάστημα κάτω από το νερό μπορεί να προκαλέσει βλάβη στους πνεύμονες που φαίνεται ώρες αργότερα: πήγαινε αμέσως σε κτηνίατρο, ακόμη κι αν το ζώο φαίνεται καλά και αναπνέει φυσιολογικά.",
         SRC["cases"], "policy", immediate=True),
    Rule("DC-ELECTRIC-CORD", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"chew\w*|bit(e|ten)?|gnaw\w*|licked|μασουλ|δαγκωσε|φαγωσε|ροκανι",
            r"cable|cord|wire|electric\w*|socket|plug|charger|power strip|lightning|καλωδι|ρευμα|πριζα|φορτιστ|ηλεκτρ|κεραυν"),
         "Chewing a live cable or any electric shock can burn the mouth and put fluid in the lungs up to a day later: go to a vet now even if the pet seems fine, and unplug the power first without touching the pet.",
         "Το μάσημα καλωδίου με ρεύμα ή κάθε ηλεκτροπληξία μπορεί να κάψει το στόμα και να βάλει υγρό στους πνεύμονες ακόμη και μια μέρα αργότερα: πήγαινε αμέσως σε κτηνίατρο ακόμη κι αν το ζώο φαίνεται καλά και κόψε πρώτα το ρεύμα χωρίς να αγγίξεις το ζώο.",
         MSD + "generalized-conditions/lightning-strike-and-electrical-injury/lightning-strike-and-electrical-injury-in-animals", "msd", immediate=True,
         exclude=r"(unplugged|not plugged|not connected|wasn'?t (plugged|connected)|μη συνδεδεμεν|δεν ηταν στην πριζα|αποσυνδεδεμεν)"),
    Rule("ALL-HYPOTHERMIA", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"hypotherm\w*|(very |ice |freezing )?cold (to the touch|body)|body (feels )?cold|feels? (very |ice )?cold|shiver\w*|frozen|frostbit\w*|κρυωσε|παγωμεν|ψυχρο σωμα|υποθερμι",
            r"weak\w*|limp|collaps\w*|confus\w*|shallow|slow (breath|heart)|not (moving|responding)|unresponsive|wet|snow|ice|outside overnight|αδυναμ|καταρρε|κατερρε|χαλαρο|αργη αναπνοη|δεν αντιδρα|βρεγμενο|χιονι|εξω ολη νυχτα"),
         "A cold, weak or wet pet that is not shivering, is confused or breathing slowly is in danger: wrap it in dry towels with a warm (never hot) water bottle against a blanket, and go to a vet now.",
         "Ζώο κρύο, αδύναμο ή βρεγμένο που δεν τρέμει, είναι μπερδεμένο ή αναπνέει αργά κινδυνεύει: τύλιξέ το σε στεγνές πετσέτες με ζεστό (όχι καυτό) μπουκάλι νερού πάνω από κουβέρτα και πήγαινε αμέσως σε κτηνίατρο.",
         SRC["cases"], "policy", immediate=True),
    Rule("DC-STING-FACE", ("dog", "cat"), "URGENT",
         _r(r"\bbee\b|\bbees\b|wasp|hornet|sting\w*|stung|ant bite|σφηκα|μελισσα|τσιμπημα εντομου|τσιμπηθηκε απο",
            r"face|muzzle|lip|mouth|tongue|throat|swollen|swelling|hives|bumps|multiple|many|προσωπο|μουσουδι|χειλι|γλωσσα|λαιμο|πρησ|κνιδωση|πολλα"),
         "Stings to the face or mouth, many stings or hives can swell the throat or trigger a severe allergic reaction: see a vet today, and go now if there is trouble breathing, collapse, pale gums, vomiting or the face swells fast.",
         "Τσιμπήματα στο πρόσωπο ή στο στόμα, πολλά τσιμπήματα ή κνίδωση μπορούν να πρηστεί ο λαιμός ή να προκαλέσουν σοβαρή αλλεργική αντίδραση: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν υπάρχει δύσκολη αναπνοή, κατάρρευση, χλωμά ούλα, εμετός ή αν πρήζεται γρήγορα το πρόσωπο.",
         MSD + "toxicology/venomous-arthropods/wasp-bee-and-ant-stings-to-animals", "msd"),
    Rule("ALL-ANAPHYLAXIS", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"allergic reaction|anaphyla\w*|face (is |has |looks )?(swelling|swollen|puffed up)|swollen (face|muzzle|lips|eyes|throat)|hives all over|αλλεργικη αντιδραση|αναφυλαξ|πρηστηκε το προσωπο|πρησμενο προσωπο|πρησμενα χειλ",
            r"breath\w*|collaps\w*|vomit\w*|pale gums|drool\w*|weak\w*|wheez\w*|can'?t swallow|αναπν|καταρρε|κατερρε|εμετ|χλωμα ουλα|σαλιαριζ|αδυναμ"),
         "A swollen face or hives with trouble breathing, vomiting, drooling, pale gums or weakness is a severe allergic reaction (anaphylaxis) that can kill within minutes: go to a vet now.",
         "Πρησμένο πρόσωπο ή κνίδωση με δύσκολη αναπνοή, εμετό, σάλιο, χλωμά ούλα ή αδυναμία είναι σοβαρή αλλεργική αντίδραση (αναφυλαξία) που μπορεί να σκοτώσει μέσα σε λεπτά: πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "dog-owners/immune-disorders-of-dogs/disorders-involving-anaphylactic-reactions-type-i-reactions-atopy-in-dogs", "msd", immediate=True),
    Rule("DC-LEPTO", ("dog", "cat"), "URGENT",
         _r(r"(swam|swimming|drank|drinking|played|walked) (in|from) (a |the )?(pond|stream|river|puddle|lake|stagnant|canal|standing water)|stagnant water|puddle water|rats?\b.{0,20}(urine|area|garden)|ποταμι|λιμνη|νερο λακκουβα|στασιμο νερο|κολυμπησε σε",
            r"letharg\w*|not eating|vomit\w*|jaundice|yellow|peeing (more|less)|no urine|drinking more|fever|muscle pain|stiff|ληθαργ|δεν τρωει|εμετ|ικτερ|κιτρινα|πινει περισσοτερο|δεν ουρει|πυρετ"),
         "A pet that is off colour after contact with stagnant water, a river or rats can have leptospirosis, which damages the kidneys and liver and can pass to people: see a vet today, wear gloves when handling urine, and go now if there are yellow gums, bleeding or no urine.",
         "Ζώο που είναι χάλια μετά από επαφή με στάσιμο νερό, ποτάμι ή αρουραίους μπορεί να έχει λεπτοσπείρωση, που βλάπτει νεφρά και ήπαρ και μεταδίδεται και σε ανθρώπους: δες κτηνίατρο σήμερα, φόρα γάντια όταν πιάνεις ούρα και πήγαινε αμέσως αν υπάρχουν κίτρινα ούλα, αιμορραγία ή δεν ουρεί.",
         MSD + "dog-owners/disorders-affecting-multiple-body-systems-of-dogs/leptospirosis-in-dogs", "policy"),
    Rule("DC-RABIES-EXPOSURE", ("dog", "cat"), "URGENT",
         _r(r"bat\b|bats\b|fox\b|foxes|jackal|wolf|wild animal|wild (dog|cat)|stray (dog|cat) bit|unknown (dog|animal) bit|bitten by (a )?(stray|wild|unknown)|νυχτεριδα|αλεπου|τσακαλι|λυκος|αδεσποτο.{0,15}δαγκωσε|αγριο ζωο",
            r"bit|bite|bitten|scratch\w*|attack\w*|fight\w*|saliva|δαγκωσ|γρατσουν|επιτεθηκε|καβγα"),
         "A bite or scratch from a bat, fox or an unknown animal needs a vet today to check the vaccination record and rabies risk, and any person who was bitten should be seen by a doctor right away.",
         "Δάγκωμα ή γρατζουνιά από νυχτερίδα, αλεπού ή άγνωστο ζώο χρειάζεται κτηνίατρο σήμερα για έλεγχο του εμβολιασμού και του κινδύνου λύσσας και όποιος άνθρωπος δαγκώθηκε πρέπει να δει γιατρό αμέσως.",
         MSD + "dog-owners/brain-spinal-cord-and-nerve-disorders-of-dogs/rabies-in-dogs", "policy"),
    Rule("ALL-BURN", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"burn\w*|scald\w*|boiling (water|oil)|hot (oil|water|pan|stove|grill|barbecue|asphalt|pavement)|caught fire|καψιμο|εγκαυμ|καψε|βραστο νερο|καυτο λαδι|ζεματ",
            r"skin|fur|paw|pad|face|mouth|body|back|belly|blister\w*|red|raw|peeling|δερμα|τριχωμα|πατουσα|προσωπο|στομα|σωμα|φουσκαλ|κοκκιν"),
         "Burns hide under fur and get worse over days: cool the area with cool (not ice) running water for several minutes, cover loosely and see a vet today, and go now if the burn is large, deep, on the face or mouth, or from a chemical or electricity.",
         "Τα εγκαύματα κρύβονται κάτω από το τρίχωμα και χειροτερεύουν μέσα σε μέρες: δρόσισε την περιοχή με δροσερό (όχι παγωμένο) τρεχούμενο νερό για αρκετά λεπτά, κάλυψέ την χαλαρά και δες κτηνίατρο σήμερα και πήγαινε αμέσως αν το έγκαυμα είναι μεγάλο, βαθύ, στο πρόσωπο ή στο στόμα ή από χημικό ή ρεύμα.",
         SRC["cases"], "policy"),
    Rule("ALL-MAGGOTS", ("dog", "cat", "rabbit", "other"), "EMERGENCY",
         _r(r"maggot\w*|fly ?strike|flystrike|fly eggs|eggs? (in|on) (the )?(fur|wound|skin)|worms? (in|crawling in|coming out of) (a |the |his |her )?(wound|sore|skin|bottom|rear)|σκουληκ\w*.{0,20}(πληγη|δερμα|τριχωμα|ουρα|πισω)|πληγη.{0,20}σκουληκ"),
         "Maggots or fly eggs on a pet spread within hours and can cause fatal shock and poisoning of the blood: go to a vet now and keep the area clean and covered with a loose cloth.",
         "Σκουλήκια ή αυγά μύγας πάνω στο ζώο πολλαπλασιάζονται μέσα σε ώρες και μπορεί να προκαλέσουν θανατηφόρο σοκ και δηλητηρίαση του αίματος: πήγαινε αμέσως σε κτηνίατρο και κράτα την περιοχή καθαρή και καλυμμένη με ένα χαλαρό πανί.",
         MSD + "integumentary-system/flies/facultative-myiasis-producing-flies", "msd", immediate=True),
    Rule("SMALL-ANTIBIOTIC", ("rabbit", "other"), "EMERGENCY",
         _r(r"clindamycin|lincomycin|erythromycin|penicillin|amoxicillin|augmentin|amoxiclav|κλινδαμυκιν|λινκομυκιν|ερυθρομυκιν|πενικιλλιν|αμοξυκιλλιν",
            r"rabbit|guinea ?pig|hamster|chinchilla|bunny|κουνελ|ινδικο χοιριδι|χαμστερ|τσιντσιλα"),
         "Oral penicillin-type antibiotics, clindamycin, lincomycin and erythromycin can kill rabbits, guinea pigs and hamsters by wiping out their gut bacteria, often within 48 hours: go to a vet now and do not give another dose.",
         "Τα από του στόματος αντιβιοτικά τύπου πενικιλλίνης, η κλινδαμυκίνη, η λινκομυκίνη και η ερυθρομυκίνη μπορούν να σκοτώσουν κουνέλια, ινδικά χοιρίδια και χάμστερ καταστρέφοντας τα βακτήρια του εντέρου, συχνά μέσα σε 48 ώρες: πήγαινε αμέσως σε κτηνίατρο και μη δώσεις άλλη δόση.",
         MSD + "exotic-and-laboratory-animals/rabbits/bacterial-and-mycotic-diseases-of-rabbits", "msd", immediate=True),
    Rule("RAB-FIPRONIL", ("rabbit", "other"), "EMERGENCY",
         _r(r"fipronil|frontline|front line|advantix|spot[- ]?on|flea (drops|treatment|spray|product|collar)|φροντλαιν|ψυλλ\w* (σταγονες|αμπουλα|σπρει)",
            r"rabbit|bunny|guinea ?pig|ferret|κουνελ|ινδικο χοιριδι|φερετ"),
         "Fipronil and many dog or cat flea products can kill rabbits and other small pets: go to a vet now, and do not bathe a weak animal yourself.",
         "Η φιπρονίλη και πολλά προϊόντα ψύλλων για σκύλους ή γάτες μπορούν να σκοτώσουν κουνέλια και άλλα μικρά ζώα: πήγαινε αμέσως σε κτηνίατρο και μη λούσεις μόνος σου αδύναμο ζώο.",
         MSD + "all-other-pets/rabbits/skin-disorders-of-rabbits", "msd", immediate=True),
    Rule("DOG-MDR1", ("dog",), "EMERGENCY",
         _r(r"collie|sheltie|shetland sheepdog|australian shepherd|aussie|old english sheepdog|border collie|herding|longhaired whippet|κολλεy|κολλευ|τσομπανοσκυλ",
            r"ivermectin|horse (wormer|paste)|cattle (wormer|pour)|loperamide|imodium|ιβερμεκτιν|λοπεραμιδ|αντιπαρασιτικο αλογου",
            r"tremor\w*|stagger\w*|wobbl\w*|drool\w*|blind\w*|can'?t (stand|walk)|collaps\w*|seizure\w*|dilated|τρεμ|τρικλιζ|σαλιαριζ|τυφλ|δεν στεκεται|καταρρε|κατερρε|σπασμ"),
         "Collies and related herding breeds can be poisoned by ivermectin (horse or cattle wormers) and loperamide at normal doses: tremors, stumbling, drooling or blindness mean go to a vet now.",
         "Οι κόλλεϋ και οι συγγενικές φυλές τσοπανόσκυλων μπορούν να δηλητηριαστούν από ιβερμεκτίνη (αντιπαρασιτικά αλόγων ή βοοειδών) και λοπεραμίδη σε κανονικές δόσεις: τρόμος, παραπάτημα, σάλιο ή τύφλωση σημαίνουν ότι πρέπει να πας αμέσως σε κτηνίατρο.",
         MSD + "pharmacology/anthelmintics/safety-of-anthelmintics-in-animals", "msd", immediate=True),
    # ---- from the MSD Immune, Infectious Diseases and Generalized Conditions pages (policy basis, vet to confirm; Greek prevalence not on MSD)
    Rule("DOG-TICK-FEVER", ("dog",), "URGENT",
         _r(r"tick|τσιμπουρ",
            r"fever|letharg\w*|swollen (glands|lymph|nodes)|limp\w*|lame\w*|not eating|no appetite|stiff|nosebleed|bruis\w*|πυρετ|ληθαργ|πρησμεν\w* (αδενες|λεμφ)|κουτσαιν|δεν τρωει|ακαμψια|αιμορραγια απο τη μυτη|μωλωπ"),
         "Fever, tiredness, swollen glands, lameness or bleeding in the weeks after a tick bite can be a tick-borne disease (ehrlichiosis, babesiosis, anaplasmosis, Lyme): see a vet today, and go now if there are pale gums, bleeding or dark urine.",
         "Πυρετός, κόπωση, πρησμένοι αδένες, κουτσαίνει ή αιμορραγία τις εβδομάδες μετά από τσιμπούρι μπορεί να είναι νόσημα από τσιμπούρι (ερλιχίωση, βαβεσίωση, αναπλάσμωση, Lyme): δες κτηνίατρο σήμερα και πήγαινε αμέσως αν υπάρχουν χλωμά ούλα, αιμορραγία ή σκούρα ούρα.",
         MSD + "infectious-diseases/rickettsial-diseases/ehrlichiosis-anaplasmosis-and-related-infections-in-animals", "policy"),
    Rule("DOG-LEISHMANIA", ("dog", "cat"), "URGENT",
         _r(r"leishman\w*|sand ?fl(y|ies)|λεισμανι|φλεβοτομ|(hair loss|scaly|crust\w*|sores?|ulcers?).{0,40}(ears?|nose|face|eyes?|lips?|αυτια|μυτη|προσωπο|ματια|χειλη)|(overgrown|long|curved|brittle) nails|μακρια νυχια",
            r"weight loss|losing weight|lost weight|thin|nosebleed|drinking (much |a lot )?more|peeing more|tired|letharg\w*|pale|limp\w*|αδυνατ|αιμορραγια απο τη μυτη|πινει περισσοτερο|κουραζ|ληθαργ|χλωμ|κουτσαιν"),
         "Skin sores or scaly hair loss on the face and ears or long nails together with weight loss, tiredness, extra thirst or nosebleeds can be leishmaniasis (a sand-fly disease common around the Mediterranean that damages the kidneys): see a vet today. People catch it from sand flies, not from touching the dog.",
         "Πληγές ή λέπια με απώλεια τριχώματος στο πρόσωπο και τα αυτιά ή μακριά νύχια μαζί με απώλεια βάρους, κούραση, αυξημένη δίψα ή αιμορραγία από τη μύτη μπορεί να είναι λεϊσμανίαση (νόσος από φλεβοτόμους, συχνή στη Μεσόγειο που βλάπτει τα νεφρά): δες κτηνίατρο σήμερα. Οι άνθρωποι τη μεταφέρουν από φλεβοτόμους, όχι από το άγγιγμα του σκύλου.",
         MSD + "infectious-diseases/leishmaniosis/leishmaniosis-in-dogs", "policy"),
    Rule("CAT-FIP", ("cat",), "URGENT",
         _r(r"swollen belly|belly (is )?(swollen|bloated|big|fluid)|fluid in (the )?(belly|chest)|cloudy eyes?|eyes? (is |are )?cloudy|fever (that )?(won'?t|doesn'?t|not) (go|respond)|not responding to antibiotics|φουσκωμενη κοιλια|υγρο στην κοιλια|θολα ματια|πυρετος που δεν περναει|δεν ανταποκρινεται στα αντιβιοτικα",
            r"weight loss|letharg\w*|not eating|fever|wobbl\w*|seizure\w*|head tilt|αδυνατ|ληθαργ|δεν τρωει|πυρετ|τρικλιζ|σπασμ"),
         "A cat with a swollen belly, cloudy eyes or a fever that does not go away with antibiotics, plus weight loss or lethargy, can have feline infectious peritonitis (FIP), which is treatable if caught early: see a vet today, and go now if breathing is laboured, gums are blue or there are fits.",
         "Γάτα με φουσκωμένη κοιλιά, θολά μάτια ή πυρετό που δεν περνά με αντιβιοτικά, μαζί με απώλεια βάρους ή λήθαργο, μπορεί να έχει λοιμώδη περιτονίτιδα της γάτας (FIP), που θεραπεύεται αν πιαστεί νωρίς: δες κτηνίατρο σήμερα και πήγαινε αμέσως αν η αναπνοή είναι δύσκολη, τα ούλα μπλε ή υπάρχουν σπασμοί.",
         MSD + "infectious-diseases/feline-infectious-peritonitis/feline-infectious-peritonitis", "policy"),
    Rule("DC-NEONATE", ("dog", "cat"), "EMERGENCY",
         _r(r"newborn|new ?born|(1|2|3|4|one|two|three|four)[- ]?(day|days|week|weeks)[- ]?old|few days old|days? old|neonat\w*|νεογνο|νεογεννητ|(μιας|δυο|τριων|τεσσαρων) (ημερων|εβδομαδων)",
            r"not (nursing|feeding|suckling|eating|moving)|won'?t (nurse|feed|suckle)|crying (a lot|constantly|nonstop)|cold|weak|limp|fast breath\w*|struggl\w*|pushed away|rejected|δεν θηλαζει|κλαιει συνεχως|κρυο|αδυναμο|χαλαρο|γρηγορη αναπνοη|το απεριψε"),
         "A newborn puppy or kitten that is not nursing, is cold, weak, crying non-stop or breathing fast can die within hours: go to a vet now and keep it warm against your body while you travel.",
         "Νεογέννητο κουτάβι ή γατάκι που δεν θηλάζει, είναι κρύο, αδύναμο, κλαίει συνεχώς ή αναπνέει γρήγορα μπορεί να πεθάνει μέσα σε ώρες: πήγαινε αμέσως σε κτηνίατρο και κράτα το ζεστό πάνω στο σώμα σου στη διαδρομή.",
         MSD + "infectious-diseases/canine-herpesvirus-infection/canine-herpesvirus-infection", "msd", immediate=True),
    Rule("BIRD-PSITTACOSIS", ("bird",), "URGENT",
         _r(r"discharge|sneez\w*|green(ish)? (droppings|poop|urates)|yellow(ish)? (droppings|poop|urates)|ruffled|fluffed|εκκριμα|φταρνισ|πρασινα περιττωματα|κιτρινα περιττωματα",
            r"letharg\w*|not eating|sleepy|quiet|breath\w*|ληθαργ|δεν τρωει|νυσταλεο|αναπνο"),
         "A bird with eye or nose discharge, ruffled feathers and green or yellow droppings can have psittacosis, which people can catch from dust and droppings: see a vet who sees birds today, wear a mask when cleaning the cage, and tell your doctor if you develop a fever or cough.",
         "Πουλί με έκκριμα από μάτια ή μύτη, φουσκωμένα φτερά και πράσινα ή κίτρινα περιττώματα μπορεί να έχει ψιττάκωση, που μεταδίδεται και σε ανθρώπους από σκόνη και περιττώματα: δες σήμερα κτηνίατρο που βλέπει πουλιά, φόρα μάσκα όταν καθαρίζεις το κλουβί και πες το στον γιατρό σου αν αποκτήσεις πυρετό ή βήχα.",
         MSD + "infectious-diseases/chlamydiosis/chlamydiosis-in-animals", "policy"),
    # ---- from the MSD pet-owner pages for rabbits, guinea pigs, hamsters, rats, chinchillas, birds, reptiles, hedgehogs (policy basis, vet to confirm)
    Rule("SMALL-HEAT", ("rabbit", "other"), "EMERGENCY",
         _r(r"rabbit|bunny|chinchilla|guinea ?pig|rat\b|κουνελ|τσιντσιλα|ινδικο χοιριδι|αρουραι",
            r"heat\w*|hot (day|weather|room|car|balcony|sun)|in the sun|overheat\w*|heatstroke|ζεστη|καυσωνα|ηλιο|υπερθερμ",
            r"pant\w*|breath\w* (fast|hard|heavy)|rapid breath\w*|stretched out|lying (flat|limp)|limp|collaps\w*|drool\w*|weak\w*|λαχανιαζ|γρηγορη αναπνοη|ξαπλωμενο|χαλαρο|καταρρε|κατερρε|σαλιαριζ|αδυναμ"),
         "Rabbits, chinchillas and guinea pigs overheat above about 27 °C: a pet that is stretched out, panting or limp in the heat is in danger within minutes: wet the ears with cool (not ice-cold) water, put it in front of a fan and go to a vet now.",
         "Τα κουνέλια, οι τσιντσιλά και τα ινδικά χοιρίδια υπερθερμαίνονται πάνω από περίπου 27 °C: ζώο που είναι ξαπλωμένο, λαχανιάζει ή είναι χαλαρό στη ζέστη κινδυνεύει μέσα σε λεπτά: βρέξε τα αυτιά με δροσερό (όχι παγωμένο) νερό, βάλ' το μπροστά σε ανεμιστήρα και πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "all-other-pets/rabbits/illness-of-rabbits", "msd", immediate=True),
    Rule("SMALL-BREATH", ("rabbit", "other"), "EMERGENCY",
         _r(r"guinea ?pig|hamster|gerbil|\bmouse\b|\bmice\b|\brat\b|\brats\b|chinchilla|ferret|ινδικο χοιριδι|χαμστερ|γερβιλ|ποντικ|αρουραι|τσιντσιλα|φερετ|κουνελ|rabbit|bunny",
            r"(difficult|trouble|hard|laboured|labored|struggl\w*|gasp\w*|open[- ]mouth)\w* (to )?breath\w*|breath\w* (is |with )?(difficult|hard|labou?red|noisy)|clicking|gasping|δυσκολια στην αναπνοη|δυσκολευεται να αναπνευσει|αναπνεει με το στομα|ασθμα"),
         "A small pet that struggles to breathe is an emergency: keep it calm and cool, handle it as little as possible and go to a vet who sees small pets now.",
         "Μικρό ζώο που δυσκολεύεται να αναπνεύσει είναι επείγον: κράτα το ήρεμο και δροσερό, πιάσ' το όσο λιγότερο γίνεται και πήγαινε αμέσως σε κτηνίατρο που βλέπει μικρά ζώα.",
         MSD + "all-other-pets/guinea-pigs/common-health-problems-of-guinea-pigs", "msd", immediate=True),
    Rule("SMALL-ANTIBIOTIC-DIARRHEA", ("rabbit", "other"), "EMERGENCY",
         _r(r"antibiotic\w*|αντιβιοτικ",
            r"diarrh\w*|loose stool|soft stool|bloat\w*|not eating|no droppings|letharg\w*|διαρρο|μαλακα κοπρανα|φουσκ|δεν τρωει|ληθαργ"),
         "Diarrhoea, bloating or not eating after an antibiotic can be fatal in rabbits, guinea pigs, hamsters and chinchillas: stop the antibiotic and go to a vet now.",
         "Διάρροια, φούσκωμα ή άρνηση φαγητού μετά από αντιβιοτικό μπορεί να είναι θανατηφόρα σε κουνέλια, ινδικά χοιρίδια, χάμστερ και τσιντσιλά: σταμάτα το αντιβιοτικό και πήγαινε αμέσως σε κτηνίατρο.",
         MSD + "all-other-pets/guinea-pigs/common-health-problems-of-guinea-pigs", "msd", immediate=True),
    Rule("HAMSTER-WETTAIL", ("other",), "EMERGENCY",
         _r(r"hamster|wet tail|wettail|gerbil|χαμστερ|γερβιλ",
            r"wet (tail|bottom|rear|rump)|soiled (rear|bottom)|rectal prolapse|bulging eye|eye (is )?bulging|(diarrh\w*|watery).{0,60}(hunched|lethargic|not eating|limp|bloat\w*)|(hunched|lethargic|not eating|limp|bloat\w*).{0,60}(diarrh\w*|watery)|υγρη ουρα|προπτωση|διαρρο.{0,60}(σκυφτο|ληθαργ|δεν τρωει)"),
         "A hamster or gerbil with a wet bottom or watery diarrhoea, a hunched posture, not eating, a bulging eye or a protruding rectum can die within a day or two: go to a vet now and give no antibiotics yourself.",
         "Χάμστερ ή γερβίλος με βρεγμένο πίσω μέρος ή υδαρή διάρροια, σκυφτή στάση, άρνηση φαγητού, εξόφθαλμο ή προβάλλον έντερο μπορεί να πεθάνει μέσα σε μία με δύο μέρες: πήγαινε αμέσως σε κτηνίατρο και μη δώσεις αντιβιοτικά μόνος σου.",
         MSD + "all-other-pets/hamsters/disorders-and-diseases-of-hamsters", "msd", immediate=True),
    Rule("BIRD-FUMES", ("bird",), "EMERGENCY",
         _r(r"non-?stick|teflon|ptfe|air ?fryer|toaster oven|waffle|self-?clean\w*|aerosol|spray|air freshener|perfume|hairspray|hair spray|burn\w* plastic|smoke|fumes|candle|αντικολλητικ|σπρει|αρωμα|καπνο|αναθυμιασ",
            r"bird|budgie|parrot|cockatiel|canary|finch|lovebird|πουλι|παπαγαλ|καναρινι"),
         "Fumes from overheated non-stick pans or appliances, aerosols or smoke can kill birds within minutes: move the bird to fresh air at once and go to a vet who sees birds now, even if it looks well.",
         "Αναθυμιάσεις από υπερθερμασμένα αντικολλητικά σκεύη ή συσκευές, σπρέι ή καπνός μπορούν να σκοτώσουν πουλιά μέσα σε λεπτά: βγάλε αμέσως το πουλί σε καθαρό αέρα και πήγαινε σε κτηνίατρο που βλέπει πουλιά, ακόμη κι αν φαίνεται καλά.",
         MSD + "bird-owners/disorders-and-diseases-of-birds/disorders-affecting-multiple-body-systems-of-pet-birds", "msd", immediate=True,
         exclude=r"smoke.?free|no smoke|without smoke|χωρις καπνο"),
    Rule("BIRD-METAL-NEURO", ("bird", "other"), "EMERGENCY",
         _r(r"tremor\w*|seizure\w*|clenched (feet|claws)|incoordinat\w*|falling off|τρομος|σπασμ|κλειστα νυχια",
            r"bird|budgie|parrot|cockatiel|grey|gray|canary|finch|lovebird|gerbil|πουλι|παπαγαλ|καναρινι"),
         "Tremors, fits, clenched feet or poor coordination in a bird (or a gerbil that chewed old metal objects) can mean lead or zinc poisoning or, in African greys, low calcium: go to a vet who sees birds now.",
         "Τρόμος, σπασμοί, σφιγμένα πόδια ή κακός συντονισμός σε πουλί (ή γερβίλο που μάσησε παλιά μεταλλικά αντικείμενα) μπορεί να σημαίνει δηλητηρίαση από μόλυβδο ή ψευδάργυρο ή, στους αφρικανικούς γκρίζους παπαγάλους, χαμηλό ασβέστιο: πήγαινε αμέσως σε κτηνίατρο που βλέπει πουλιά.",
         MSD + "bird-owners/disorders-and-diseases-of-birds/disorders-affecting-multiple-body-systems-of-pet-birds", "msd", immediate=True),
    Rule("REP-TORTOISE-SERIOUS", ("reptile",), "EMERGENCY",
         _r(r"tortoise|turtle|terrapin|χελωνα|νερογελωνα",
            r"open[- ]mouth|mouth open|bubbl\w*|nose (running|bubbles)|can'?t walk|dragging|limp|tremor\w*|spasm\w*|shell (crack\w*|broke\w*|crushed)|crack in (the )?shell|open wound|swollen (and )?closed eyes|eyes? (swollen|closed)|prolaps\w*|(hot|heat) lamp.{0,20}burn|ανοιχτο στομα|φυσαλιδ|δεν περπατα|σερνεται|τρομος|σπασμ|ραγισμα|σπασμενο καβουκι|πρησμενα ματια|προπτωση|εγκαυμα"),
         "A tortoise or turtle with an open mouth, bubbles from the nose, a cracked shell, trouble walking, tremors, swollen closed eyes or a prolapse needs a reptile vet now: keep it warm (in the upper part of its normal range), cover an open wound with a clean damp cloth and do not force-feed.",
         "Χελώνα με ανοιχτό στόμα, φυσαλίδες από τη μύτη, ραγισμένο καβούκι, δυσκολία στο περπάτημα, τρόμο, πρησμένα κλειστά μάτια ή πρόπτωση χρειάζεται αμέσως κτηνίατρο ερπετών: κράτα την ζεστή (στο πάνω μέρος του φυσιολογικού εύρους), κάλυψε ανοιχτή πληγή με καθαρό υγρό πανί και μην την αναγκάσεις να φάει.",
         MSD + "all-other-pets/reptiles/emergencies-of-reptiles", "policy", immediate=True),
    Rule("HEDGEHOG-TORPOR", ("other",), "EMERGENCY",
         _r(r"hedgehog|σκαντζοχοιρ",
            r"cold|torpor|unresponsive|not moving|limp|won'?t wake|slow breath\w*|labou?red breath\w*|noisy breath\w*|κρυο|δεν κινειται|δεν ξυπναει|αργη αναπνοη|δυσκολη αναπνοη"),
         "A cold, unresponsive or hibernating-looking hedgehog, or one with noisy breathing, is in danger: warm it slowly and gently against your body and go to a vet now; do not let it stay cold to hibernate.",
         "Σκαντζόχοιρος κρύος, που δεν αντιδρά ή φαίνεται να πέφτει σε χειμερία νάρκη ή αναπνέει θορυβωδώς κινδυνεύει: ζέστανέ τον σιγά σιγά πάνω στο σώμα σου και πήγαινε αμέσως σε κτηνίατρο· μην τον αφήσεις κρύο για να πέσει σε χειμερία νάρκη.",
         MSD + "exotic-and-laboratory-animals/four-toed-african-pygmy-hedgehogs/diseases-of-four-toed-african-pygmy-hedgehogs", "policy", immediate=True),
    # ---- from MSD anemia (pet-owner), wound-management and hyperthyroid pages (policy basis, levels inferred, vet to confirm)
    Rule("DC-ANEMIA-JAUNDICE-WEAK", ("dog", "cat"), "EMERGENCY",
         _r(r"yellow\w* (skin|gums?|eyes?|whites)|(skin|gums?|eyes?|whites).{0,20}yellow|jaundic|icter|ικτερ|κιτριν\w* (ουλα|δερμα|ματια)",
            r"collaps\w*|faint\w*|can'?t (stand|get up|walk)|very weak|so weak|racing heart|heart (is )?rac\w*|fast heart|panting at rest|καταρρε|κατερρε|δεν σηκωνεται|πολυ αδυνατ"),
         "Yellow gums, skin or eyes together with collapse, severe weakness or a racing heart can be destruction of red blood cells (for example immune-mediated anaemia), which can be fatal without rapid treatment: go to a vet now and give no human painkillers.",
         "Κίτρινα ούλα, δέρμα ή μάτια μαζί με κατάρρευση, σοβαρή αδυναμία ή γρήγορο καρδιακό ρυθμό μπορεί να σημαίνουν καταστροφή ερυθρών αιμοσφαιρίων (π.χ. ανοσομεταβολική αναιμία) και είναι επείγον: πήγαινε αμέσως σε κτηνίατρο και μην δώσεις ανθρώπινα παρηγορητικά.",
         MSD + "dog-owners/blood-disorders-of-dogs/anemia-in-dogs", "policy", immediate=True),
    Rule("DC-BITE-ANY", ("dog", "cat", "rabbit", "other"), "URGENT",
         _r(r"bitten|was bit\b|got bit\b|bite wound|bite mark|dog bit|cat bit|fight with (a |another )?(dog|cat)|attacked by (a |another )?(dog|cat|fox|jackal)|δαγκωσε|δαγκωθηκε|δαγκωσαν|δαγκωμα|επιτεθηκε απο|επιθεση απο"),
         "Any bite should be seen by a vet: a small puncture can hide deep infection, broken ribs or organ damage, and cat bites infect easily. Press on bleeding, rinse with plain water or saline, put nothing else on it, and go today.",
         "Κάθε δάγκωμα χρειάζεται κτηνίατρο: ένα μικρό τρύπημα μπορεί να κρύβει βαθιά μόλυνση, σπασμένα πλευρά ή βλάβη σε όργανα, και τα δαγκώματα γάτας μολύνονται εύκολα. Πίεσε αν αιμορραγεί, ξέπλυνε με σκέτο νερό ή φυσιολογικό ορό, μην βάλεις τίποτα άλλο και πήγαινε σήμερα.",
         "https://www.msdvetmanual.com/emergency-medicine-and-critical-care/wound-management-in-small-animals/initial-wound-management-in-small-animals", "policy", exclude=r"(not|n'?t|never) (been |got |get )?bit|snake|viper|spider|insect|\bbee\b|mosquito|flea|tick|φιδι|οφη"),
    Rule("DC-YOUNG-GI", ("dog", "cat"), "URGENT",
         _r(r"puppy|\bpup\b|kitten|κουταβ|γατακ|\b([1-9]|1[0-9]|2[0-4]) ?[- ]?(weeks?|εβδομαδ)[- ]?(old|ων|ης)?|\b([1-5]) ?[- ]?(months?|μην\w*)[- ]?(old|ων|ης)?|young|new ?born",
            r"vomit|diarrh|letharg|not eating|poor appetite|reduced appetite|soft stool|yellow stool|εμετ|διαρρο|ληθαργ|δεν τρωει|αδυναμ|μαλακα κοπρανα"),
         "A puppy or kitten with vomiting, diarrhoea, lethargy or poor appetite can crash within hours (low blood sugar, dehydration, parvovirus, parasites): see a vet today.",
         "Ένα κουτάβι ή γατάκι με εμετό, διάρροια, λήθαργο ή κακή όρεξη μπορεί να χειροτερέψει μέσα σε ώρες (χαμηλό σάκχαρο, αφυδάτωση, παρβοϊός, παράσιτα): δες κτηνίατρο σήμερα.",
         SRC["dog_gi"] + " | " + SRC["cases"], "policy"),
]

# exposure rules: ignore clauses where the owner says it did not happen
def _fold_pat(p):
    """Greek letters in a pattern get the same treatment as the text (lower-case, no accents, final sigma -> sigma),
    otherwise an alternative written with a final sigma or an accent can never match. ASCII (regex syntax) is untouched."""
    return "".join(norm(c) if "\u0370" <= c <= "\u03ff" else c for c in p)


for _rule in RULES:
    _rule.all_of = tuple(re.compile(_fold_pat(x.pattern), x.flags) for x in _rule.all_of)
    if _rule.exclude:
        _rule.exclude = _fold_pat(_rule.exclude)

_GUARDED = {"DOG-XYLITOL", "DOG-GRAPES", "CAT-LILY", "CAT-FLOWERS", "CAT-PARACETAMOL", "DC-NSAID", "DC-ANTIFREEZE", "DC-RODENTICIDE",
            "DC-RODENTICIDE-BLEED", "DC-SNAKEBITE", "DC-METALDEHYDE", "DC-POISON-BAIT", "CAT-DOGFLEA", "DC-BATTERY", "DC-CORROSIVE",
            "DC-ALCOHOL", "DC-OTC-STIMULANT", "DC-RX-HUMAN", "DC-FLUOROURACIL", "DC-PLANT-SEVERE", "DC-PLANT-OTHER", "DOG-BREADDOUGH",
            "BIRD-AVOCADO", "DC-SALT"}
_GUARDED |= {"DOG-MACADAMIA", "DC-ONION-GARLIC", "DC-MOTHBALL", "CAT-ESSENTIAL-OIL", "DOG-ESSENTIAL-OIL", "DC-RODENTICIDE-NEURO", "DC-THC", "DC-TOPICAL-HUMAN", "DC-CLEANER-POD"}
for _rule in RULES:
    if _rule.id in _GUARDED:
        _rule.neg_guard = True

# rules (plus the chocolate calculator, id DC-CHOCOLATE) that mean "the pet was exposed to a poison": used for the hand-off card
TOXIN_IDS = _GUARDED | {"DC-CHOCOLATE"}
BAIT_IDS = {"DC-POISON-BAIT", "DC-METALDEHYDE", "DC-RODENTICIDE"}


# ---- chocolate (dose-aware, from MSD figures) --------------------------------
_CHOC = re.compile(r"chocolat|\bchoc\b|cocoa|brownie|hershey|cadbury|lindt|milka|toblerone|kit ?kat|snickers|"
                   r"σοκολατ|κακαο|μπρουνι|μιλκα|παυλιδη|γαλαξια|λινδτ")
_TRACE = re.compile(r"crumb|lick|tiny|trace|a taste|just a (bite|bit|taste)|small bit|ψιχουλ|γλειψ|μια δαγκι|μικρο κομματακι")
_GRAMS = re.compile(r"(\d+(?:[.,]\d+)?) ?(?:g\b|gr\b|gram|γρ\b|γραμ)")
_WEIGHT = re.compile(r"(\d+(?:[.,]\d+)?) ?(?:kg|kilo|κιλ)")
# methylxanthines in mg per gram of chocolate (MSD)
_MGPG = (("cocoa powder|κακαο σκονη|σκονη κακαο", 28.5), ("baker|unsweetened|κουβερτουρα|ανευ ζαχαρης", 15.5),
         ("dark|semi|bitter|μαυρη|bittersweet|hershey", 5.5), ("milk|γαλακτος|milka|μιλκα|cadbury", 2.3),
         ("white|λευκη", 0.1))
_BIG = re.compile(r"\bbag\b|\bbox\b|\bpacks?\b|packets?|\bbars\b|whole|\blots?\b|many|several|πακετ|κουτι|ολοκληρ|πολλ")
_MID = re.compile(r"\bbar\b|squares?|pieces?|cake|brownie|πλακα|κομματ|τουρτα")


def _weight_kg(text, profile):
    w = (profile or {}).get("weight")
    try:
        w = float(str(w).replace(",", ".")) if w not in (None, "", 0) else None
    except ValueError:
        w = None
    if w and w > 0:
        return w
    m = _WEIGHT.search(text)
    return float(m.group(1).replace(",", ".")) if m else None


def chocolate_assessment(text, weight_kg=None):
    """Return (level or '', mg_per_kg or None, note) from normalized text."""
    if not _CHOC.search(text):
        return "", None, ""
    grams = None
    gm = _GRAMS.search(text)
    explicit = bool(gm)
    if gm:
        grams = float(gm.group(1).replace(",", "."))
    elif _TRACE.search(text):
        return "", None, "trace"
    elif _BIG.search(text):
        grams = 100.0
    elif _MID.search(text):
        grams = 40.0
    mgpg = 5.5  # unknown type: assume dark chocolate (the safer guess)
    for pat, val in _MGPG:
        if re.search(pat, text):
            mgpg = val
            break
    if grams is None or not weight_kg:
        return "URGENT", None, "unknown amount or weight"
    mgkg = grams * mgpg / weight_kg
    if explicit:
        lvl = "EMERGENCY" if mgkg >= 40 else "URGENT" if mgkg >= 20 else ""
    else:
        lvl = "EMERGENCY" if mgkg >= 40 else "URGENT"
    return lvl, mgkg, "explicit amount" if explicit else "assumed amount"


# ---- negation guard (exposure rules only) -----------------------------------------------------------------------
# "he did NOT eat the grapes" must not fire the grape rule, but "I'm not sure he ate grapes" must. A negated clause runs
# from the negation word to the next punctuation or "but/and", so a second, real exposure after "and" is still seen.
_NEG = re.compile(
    r"(?:\b(?:did ?n[o']?t|didn'?t|does ?n[o']?t|doesn'?t|do not|don'?t|never|not|no|without|hasn'?t|has not|haven'?t|have not|wasn'?t|was not|isn'?t|is not)\b|"
    r"\bδεν\b|\bουτε\b|\bμην\b)\s+(?:\w+\s+){0,2}?"
    r"(?:eat\w*|ate|eaten|swallow\w*|chew\w*|touch\w*|get|got|lick\w*|drank|drink\w*|take|took|taken|have|had|been|bit\w*|"
    r"εφαγ\w*|φαει|φαγε|καταπι\w*|μασησ\w*|ακουμπ\w*|ηπι\w*|γλειψ\w*|πηρε|δαγκ\w*|βρηκε)"
    r"[^.;,\n]*?(?=\bbut\b|\band\b|\bhowever\b|\bthough\b|\bthen\b|αλλα|\bκαι\b|ομως|[.;,\n]|$)")
_UNSURE = re.compile(r"\bsure\b|certain|\bknow\b|ξερω|βεβαι|\bthink|believe|νομιζ|maybe|might|perhaps|possib|ισως|wonder|worried|afraid|φοβαμαι|can'?t tell|cannot tell|not clear")


def strip_negated(text):
    """Remove clauses in which the owner says the exposure did NOT happen. Uncertain wording is kept (safer)."""
    def repl(m):
        window = text[max(0, m.start() - 20):m.end()]
        return " " if not _UNSURE.search(window) else m.group(0)
    return _NEG.sub(repl, text)


def evaluate(species_key, user_texts, profile=None):
    """Check every owner message of the current chat. Returns Result."""
    text = norm(" \n ".join(t for t in (user_texts or []) if t))
    sp = species_key or "other"
    hits = []
    if not text.strip():
        return Result()
    tg = strip_negated(text)
    for r in RULES:
        if r.species and sp not in r.species:
            continue
        t = tg if r.neg_guard else text
        if r.exclude and re.search(r.exclude, t):
            continue
        if all(p.search(t) for p in r.all_of):
            hits.append(Hit(r.id, r.level, r.en, r.el, r.source, r.basis, r.immediate))
    if sp in ("dog", "cat"):
        lvl, mgkg, note = chocolate_assessment(tg, _weight_kg(text, profile))
        if lvl:
            if mgkg is not None:
                en = (f"Chocolate: estimated {mgkg:.0f} mg/kg of methylxanthines ({note}); "
                      "about 20 mg/kg can cause mild signs and 40 mg/kg or more can affect the heart.")
                el = (f"Σοκολάτα: εκτιμώμενα {mgkg:.0f} mg/kg μεθυλοξανθινών ({note}); "
                      "περίπου 20 mg/kg προκαλούν ήπια συμπτώματα και 40 mg/kg ή περισσότερο μπορεί να επηρεάσουν την καρδιά.")
            else:
                en = "Chocolate eaten and the amount or weight is unknown: call a vet today and say how much and what kind."
                el = "Έφαγε σοκολάτα και η ποσότητα ή το βάρος είναι άγνωστα: κάλεσε κτηνίατρο σήμερα και πες πόση και τι είδους."
            hits.append(Hit("DC-CHOCOLATE", lvl, en, el, SRC["chocolate"], "msd", lvl == "EMERGENCY"))
    res = Result(hits=hits)
    if hits:
        res.level = max((h.level for h in hits), key=lambda x: RANK[x])
    return res




# Owner statements that are enough on their own to say "go now" without the six-question cycle: a collapsed or
# blue-gummed animal, a confirmed toxin, a blocked airway or urethra, an eye out of its socket... Hand-picked from the
# EMERGENCY rules (policy basis, vet to confirm). Everything else stays under the AI's question cycle; the floor still
# raises the final verdict for those.
ACT_NOW = frozenset({
    "ALL-COLLAPSE", "ALL-GUMS", "ALL-BREATH", "CAT-OPENMOUTH", "ALL-SEIZURE-LONG", "ALL-TRAUMA-BLEED", "ALL-HEATSTROKE",
    "ALL-DROWNING", "ALL-SMOKE", "ALL-ANAPHYLAXIS", "DC-ELECTRIC-CORD", "DC-CHOKING", "DC-SNAKEBITE", "DOG-GDV",
    "CAT-URETHRAL", "DOG-URETHRAL", "DOG-XYLITOL", "DOG-GRAPES", "CAT-LILY", "CAT-PARACETAMOL", "DC-ANTIFREEZE",
    "DC-RODENTICIDE-NEURO", "DC-POISON-BAIT", "DC-METALDEHYDE", "DC-FLUOROURACIL", "DC-BATTERY", "CAT-DOGFLEA",
    "DC-MUSHROOM", "CAT-ESSENTIAL-OIL", "BIRD-FUMES", "BIRD-AVOCADO", "DC-BITE-NECK-SWELL", "DC-FRACTURE-OPEN",
    "ALL-EYE-PROPTOSIS", "DC-EYE-PROPTOSIS", "ALL-EYE-PENETRATING", "DC-CHOCOLATE", "BIRD-TAILBOB",
})


def act_now_hits(result):
    """EMERGENCY hits that justify skipping the question cycle."""
    return [h for h in result.hits if h.level == "EMERGENCY" and h.id in ACT_NOW]


def act_now_reply(hits, lang="en"):
    """Fixed emergency message built from the rules, shown without waiting for the AI."""
    top = hits[:3]
    if lang == "el":
        head = "🚨 **ΠΗΓΑΙΝΕ ΑΜΕΣΩΣ ΣΕ ΚΤΗΝΙΑΤΡΕΙΟ**"
        body = "\n".join(f"- {h.el}" for h in top)
        foot = ("Τηλεφώνησε στο κτηνιατρείο στον δρόμο ώστε να ετοιμαστούν. Μην δώσεις φαγητό, νερό ή φάρμακο αν δεν στο πει ο κτηνίατρος. "
                "Μπορείς να συνεχίσεις να γράφεις εδώ, αλλά μην περιμένεις εμένα για να φύγεις.")
    else:
        head = "🚨 **GO TO A VET CLINIC IMMEDIATELY**"
        body = "\n".join(f"- {h.en}" for h in top)
        foot = ("Call the clinic on the way so they can prepare. Do not give food, water or any medicine unless the vet tells you to. "
                "You can keep writing here, but do not wait for me before you leave.")
    return f"{head}\n\n{body}\n\n{foot}"


def outranks(floor_level, ai_level):
    """True when the rule floor is higher than the AI's final level."""
    return bool(floor_level) and RANK.get(floor_level, 0) > RANK.get(ai_level or "SELF_CARE", 0)


def prompt_hints(result):
    """Short English block appended to the nurse system prompt."""
    if not result.hits:
        return ""
    lines = ["SAFETY RULE ALERTS (deterministic checks on the owner's words; ask about these first, "
             "and do not end below the highest level shown):"]
    for h in sorted(result.hits, key=lambda x: -RANK[x.level]):
        flag = " [clear immediate danger: you may skip the question cycle]" if h.immediate and h.level == "EMERGENCY" else ""
        lines.append(f"- {h.id} -> at least {h.level}: {h.en}{flag}")
    return "\n".join(lines)


def floor_note(result, lang="en"):
    """One sentence appended to the reply when the floor overrides the AI level."""
    if not result.hits:
        return ""
    top = sorted(result.hits, key=lambda x: -RANK[x.level])[0]
    head = "⚠️ Κανόνας ασφαλείας: " if lang == "el" else "⚠️ Safety rule: "
    return head + (top.el if lang == "el" else top.en)
