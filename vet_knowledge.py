"""Short, paraphrased owner-level facts from the MSD Veterinary Manual pet-owner pages, injected into the nurse prompt
only for the topics a chat actually mentions (keeps the prompt small). Content is paraphrased, never quoted; each topic
keeps its MSD page for traceability. Pure functions, no Streamlit: unit-testable.

ALWAYS holds the handful of "do / don't" rules the nurse must follow in every poisoning or first-aid conversation.
"""
import re

from vet_rules import MSD, norm

ALWAYS = (
    "FIRST-AID GUARDRAILS (from MSD pet-owner pages): never tell the owner to make the pet vomit at home or give human medicines "
    "unless a vet or poison line says so; for suspected poisoning, severe pain, uncontrolled bleeding, breathing trouble, eye injury or a "
    "suspected broken bone, tell the owner to call the vet and go immediately; do not give instructions that delay getting to a vet."
)

# (topic id, keyword regex on normalised EN+EL text, species or () for all, fact text, MSD page)
TOPICS = [
    ("chocolate", r"chocolat|cocoa|σοκολατ|κακαο", ("dog", "cat"),
     "Chocolate: the toxins are theobromine and caffeine; cocoa powder and baker's chocolate are the strongest, then dark, then milk; white is negligible. "
     "Signs usually start 6 to 12 hours later and can last up to 72 hours: restlessness, vomiting, heavy urination, tremors, racing or irregular heart, seizures. "
     "Dose and type decide the risk, so ask the kind, the amount and the pet's weight.",
     "special-pet-topics/poisoning/food-hazards"),
    ("grapes", r"grape|raisin|sultana|σταφυλ|σταφιδ", ("dog",),
     "Grapes and raisins: no safe dose is known; kidney failure can follow 1 to 3 days later, preceded by vomiting within 6 to 12 hours. "
     "Treatment is most effective if started within the first hours, so treat any ingestion as urgent.",
     "special-pet-topics/poisoning/food-hazards"),
    ("xylitol", r"xylitol|ξυλιτολ|sugar.?free|χωρις ζαχαρη", ("dog",),
     "Xylitol (sugar-free gum, sweets, some peanut butters): blood sugar can crash within 30 minutes, later liver damage 1 to 2 days on. "
     "Do not induce vomiting at home; activated charcoal does not bind it. Needs a veterinary hospital.",
     "special-pet-topics/poisoning/food-hazards"),
    ("dough", r"\bdough\b|ζυμη|ζυμαρι", ("dog",),
     "Raw bread dough expands in the stomach and ferments to alcohol: look for retching with a swollen belly, staggering, weakness, seizures. It can twist the stomach.",
     "special-pet-topics/poisoning/food-hazards"),
    ("macadamia", r"macadamia|μακαντ", ("dog",),
     "Macadamia nuts: vomiting, weakness, tremors, wobbling and fever within about 12 hours; usually resolves in 12 to 48 hours but a vet call is sensible.",
     "special-pet-topics/poisoning/food-hazards"),
    ("avocado", r"avocado|guacamole|αβοκαντο", ("bird", "rabbit", "dog", "cat"),
     "Avocado: dangerous to birds in particular (heart damage and death within about 48 hours, breathing difficulty, fluffed up, swollen chest); also harms other mammals.",
     "special-pet-topics/poisoning/food-hazards"),
    ("alcohol", r"alcohol|\bbeer|\bwine|vodka|αλκοολ|μπυρα|κρασι", ("dog", "cat"),
     "Alcohols (drinks, rubbing alcohol, some flea sprays, windscreen fluid): signs start in 30 to 60 minutes: vomiting, drooling, wobbling, tremors, slow breathing, low temperature.",
     "special-pet-topics/poisoning/household-hazards"),
    ("corrosives", r"bleach|\bdrain (cleaner|opener)|toilet (cleaner|bowl)|dishwasher (tab|det|pod|powd)|oven cleaner|corrosive|caustic|χλωριν|καθαριστικ|αποφρακτικ", ("dog", "cat"),
     "Cleaners and bleach: for a recent mouthful give plain water or milk, do NOT induce vomiting and do not give charcoal. Alkaline products (drain openers, dishwasher powder) "
     "may not hurt on contact and burns can appear up to 12 hours later. Flush skin and eyes with water (eyes for at least 20 minutes) on the way to the vet.",
     "special-pet-topics/poisoning/household-hazards"),
    ("fabric_softener", r"dryer sheet|fabric soft|potpourri|softener|μαλακτικ", ("cat", "dog"),
     "Fabric softeners, dryer sheets and liquid potpourri (cationic detergents) ulcerate the mouth within 6 to 12 hours; cats are at higher risk because they groom. Do not induce vomiting.",
     "special-pet-topics/poisoning/household-hazards"),
    ("battery", r"batter|μπαταρι", ("dog", "cat"),
     "Swallowed batteries: do not induce vomiting if the casing may be punctured. Button and lithium cells can lodge in the oesophagus and burn through; go to a vet as soon as possible.",
     "special-pet-topics/poisoning/household-hazards"),
    ("nsaid_paracetamol", r"ibuprofen|nurofen|naproxen|aspirin|paracetamol|acetaminophen|panadol|depon|ιβουπροφ|ασπιριν|παρακεταμολ", ("dog", "cat"),
     "Human painkillers: ibuprofen is dangerous to dogs (stomach ulcers, kidney damage) and cats react at about half the dog dose; aspirin can be fatal to cats in large doses; "
     "naproxen can cause black stools and bleeding; paracetamol is far more dangerous to cats (brown or blue gums, fast breathing, swollen face or paws within hours). Time matters.",
     "special-pet-topics/poisoning/poisoning-from-human-over-the-counter-drugs"),
    ("decongestant_vitamin", r"sudafed|pseudoephedrine|decongest|nasal spray|vitamin|\biron (pill|tab|suppl)|multivit|βιταμιν|σιδηρ", ("dog", "cat"),
     "Cold remedies and supplements: pseudoephedrine and nasal-spray decongestants cause racing heart, tremors and agitation for 1 to 4 days; iron causes vomiting in about 6 hours "
     "then a deceptive quiet period before liver damage 12 to 24 hours later; vitamin D causes kidney failure 1 to 2 days later.",
     "special-pet-topics/poisoning/poisoning-from-human-over-the-counter-drugs"),
    ("rx_drugs", r"antidepress|sertraline|prozac|beta.?block|blood pressure|sleeping|zolpidem|baclofen|adhd|ritalin|adderall|fluorouracil|psoriasis|χαπι|αντικαταθλιπτ|υπνωτικ|ηρεμιστικ", ("dog", "cat"),
     "Human prescription drugs: antidepressants, beta-blockers, calcium-channel blockers and ACE inhibitors can cause heart-rhythm and blood-pressure problems, seizures or coma, and some act after a delay; "
     "5-fluorouracil cream can be fatal even at low doses; psoriasis cream (calcipotriene) raises calcium and causes kidney failure 1 to 3 days later; some sleeping pills make dogs agitated instead of sleepy.",
     "special-pet-topics/poisoning/poisonings-from-human-prescription-drugs"),
    ("plants", r"\bplants?\b|\bleaf|leaves|flower|oleander|yew|azalea|lily|sago|cycad|philodendron|dieffenbachia|cannabis|marijuana|φυτο|φυλλ|λουλουδ|πικροδαφν|κρινο|κανναβ", ("dog", "cat"),
     "Plants: oleander, yew, lily of the valley, azalea, castor bean, mistletoe and autumn crocus can be fatal (heart, gut, breathing); true lilies and daylilies cause kidney failure in cats 2 to 4 days later "
     "(early vomiting, depression, no appetite within 12 hours); philodendron and dumb cane swell the mouth and throat; cycads (sago, coontie) cause persistent vomiting and liver problems; "
     "cannabis causes wobbling, dilated pupils, slow heart and tremors. Always ask for the plant's name or a photo.",
     "special-pet-topics/poisoning/plants-poisonous-to-animals"),
    ("snakebite", r"snake|viper|adder|φιδι|οχια", ("dog", "cat"),
     "Snakebite: venom acts from the first minutes and the first 2 hours are the most dangerous for pit-viper-type bites; antivenom works best within about 6 hours. "
     "Keep the pet calm and still and go to a vet immediately; do NOT ice, cut, suck, use a tourniquet or heat, and do not wait to see what happens. "
     "A mild-looking bite does not rule out venom. A photo of the snake is fine, but never delay or damage its head.",
     "special-pet-topics/poisoning/snakebite"),
    ("insecticide", r"\bflea|\btick\b|permethrin|pyrethr|insecticide|pesticide|organophosph|carbamate|spot.?on|ψυλλ|τσιμπουρ|εντομοκτον|περμεθριν", ("dog", "cat"),
     "Flea and tick products and insecticides: cats are especially sensitive, and dog products used on a cat can cause drooling, tremors and seizures within hours. "
     "For skin exposure bathe gently with mild dish detergent and lukewarm water; avoid vegetable oils; do not induce vomiting unless a vet says so. "
     "Drooling, small pupils, diarrhoea, breathing trouble, tremors or seizures after exposure need a vet immediately.",
     "special-pet-topics/poisoning/insecticide-poisoning"),
    ("metaldehyde", r"\bslug|\bsnail|metaldehyde|σαλιγκαρ", ("dog", "cat"),
     "Slug and snail bait (metaldehyde): tremors, high temperature, rapid breathing, seizures and rhythmic eye movements; about 3 ounces of bait can poison a 30-pound dog. "
     "Needs intensive hospital treatment (often 4+ days); the outcome depends on controlling fever and seizures.",
     "special-pet-topics/poisoning/metaldehyde-poisoning"),
    ("salt", r"\bsalt\b|αλατι", ("dog", "cat"),
     "Salt poisoning: follows a large intake of salt with too little water (salt dough, rock salt). Rehydrate carefully: large amounts of water at once can worsen brain swelling, "
     "so offer small sips at short intervals; death rates can exceed 50 percent, so call a vet immediately.",
     "special-pet-topics/poisoning/salt-poisoning"),
    ("heat", r"\bheat|\bhot\b|overheat|panting|θερμοπληξ|ζεστ|λαχανιαζ", (),
     "Heatstroke: hot skin, rapid panting, drooling, vomiting, wobbling, collapse. First aid on the way to the vet: move to shade, cool the head and body with cool (not ice-cold) water, wet towels, a fan, "
     "belly and groin first; offer small sips of water once temperature is falling; do not immerse in cold water. Normal dog and cat temperature is about 38.6 to 38.9 C (101.5 to 102 F).",
     "special-pet-topics/emergencies/what-to-do-in-a-dog-or-cat-emergency"),
    ("choking_bleeding_burn", r"\bchok|bleed|wound|\bburn|hypotherm|πνιγ|αιμορραγ|πληγ|εγκαυμ", ("dog", "cat"),
     "First aid: choking (cough, drool, pawing the mouth): look but never put fingers deep in the mouth, give firm thumps between the shoulder blades or quick side compressions of the rib cage. "
     "Bleeding: firm pressure with a pad, bandage firmly but not tight, add layers on top instead of removing a soaked pad. Burns: flush with cool water and cover with a non-stick dressing; "
     "face, chemical and electrical burns need immediate care. Hypothermia: dry the pet and wrap with warm (not hot) water bottles; shivering is not reliable. "
     "Do not remove objects stuck in the chest, skull or belly; do not muzzle dogs with chest injuries or short noses.",
     "special-pet-topics/emergencies/what-to-do-in-a-dog-or-cat-emergency"),
    ("small_pets", r"guinea ?pig|cavy|hamster|ferret|\brat\b|\bmouse\b|chinchilla|ινδικ\w* χοιριδι|χοιριδι\w* ινδι|χαμστερ|κουναβι", ("other",),
     "Small pets (MSD): a guinea pig, hamster or ferret that stops eating needs a prompt vet check (they dehydrate and can die suddenly); breathing difficulty is an emergency; "
     "pneumonia is a leading cause of death in guinea pigs (nose/eye discharge, sneezing, wheeze, weak). Guinea pigs need vitamin C every day (at least 10 mg per kg, about 10 mg each; 30 mg per kg if pregnant): "
     "deficiency causes swollen joints, limping, bleeding and can be fatal. Hamster diarrhoea (wet tail) is dangerous, worst in young animals; never give leftover or non-prescribed antibiotics. "
     "Ferrets hide illness: go immediately for breathing trouble, no food or water for 24 hours, temperature over 40.5 C, seizures, staggering, weakness, hind-leg dragging, pale or blue gums, bloody or black tarry stool, "
     "straining without urine or stool, or choking; vomiting or diarrhoea beyond 24 hours needs a vet within the day.",
     "all-other-pets/ferrets/emergencies-of-ferrets"),
    ("bloat", r"bloat|swollen|belly|abdomen|retch|φουσκ|κοιλι|ξερομ", ("dog",),
     "Bloat (gastric dilatation-volvulus): restlessness, swollen painful belly, repeated unproductive retching, heavy drooling, fast breathing; a weak pulse or pale gums mean shock. "
     "About 25 to 30 percent of dogs die; large deep-chested breeds are at higher risk, especially after a big meal and exercise. Treat as an emergency.",
     "dog-owners/digestive-disorders-of-dogs/disorders-of-the-stomach-and-intestines-in-dogs"),
    ("parvo_gi", r"parvo|puppy|pup\b|bloody|diarrh|vomit|κουταβ|διαρρο|εμετ|αιμα", ("dog",),
     "Puppy or unvaccinated dog with vomiting and bloody diarrhoea: think parvovirus (signs 5 to 7 days after exposure; some die within hours; 68 to 92 percent survive with treatment). "
     "Puppies are vaccinated at 6 to 8, 10 to 12 and 14 to 16 weeks. The virus lingers in faeces for weeks, so keep the dog away from others. "
     "Sudden bloody diarrhoea in a previously healthy small dog can be haemorrhagic gastroenteritis and causes fast fluid loss. "
     "Repeated vomiting, belly pain, or a swallowed sock, string, needle or toy suggest obstruction and need a vet.",
     "dog-owners/digestive-disorders-of-dogs/disorders-of-the-stomach-and-intestines-in-dogs"),
    ("allium", r"onion|garlic|chive|leek|κρεμμυδ|σκορδ|πρασ", ("dog", "cat"),
     "Onion, garlic, chives and leeks (garlic about 3 to 5 times stronger than onion, powders and dried soup mixes the strongest) damage red blood cells, cats most of all. "
     "Signs appear days later: weakness, poor appetite, pale gums, yellow eyes or skin, dark urine, fast breathing. Pale gums, dark urine or collapse are an emergency.",
     "toxicology/food-hazards/garlic-and-onion-allium-spp-toxicosis-in-animals"),
    ("macadamia", r"macadamia|μακαντα?μια", ("dog",),
     "Macadamia nuts: vomiting, weakness, wobbly gait, tremors and fever within about 12 hours; usually clears in 1 to 2 days, but call a vet the same day.",
     "toxicology/food-hazards/macadamia-nut-toxicosis-in-dogs"),
    ("essential_oils", r"essential oil|tea ?tree|diffuser|αιθεριο|τι τρι", ("dog", "cat", "bird"),
     "Essential oils (tea tree, wintergreen, pennyroyal, eucalyptus, cinnamon) are most dangerous to cats and birds, by skin, mouth or inhaled mist: drooling, vomiting, wobbliness, tremors, breathing trouble. "
     "Do not induce vomiting; get the pet into fresh air and to a vet.",
     "toxicology/toxicoses-from-household-hazards/toxicoses-from-essential-oils-in-animals"),
    ("cannabis", r"cannabis|marijuana|marihuana|\bthc\b|hashish|κανναβ|μαριχουανα|χασισ", ("dog", "cat"),
     "Cannabis/THC: wobbliness, sleepiness, vomiting, leaking urine, head bobbing and sensitivity to noise within minutes to 2 hours (up to 4 if fasted); lasts 1 to 3 days. Deaths are rare, but seizures, collapse or slow breathing are an emergency.",
     "toxicology/toxicoses-from-human-recreational-drugs/toxicosis-in-dogs-and-cats-from-tetrahydrocannabinol-thc"),
    ("smoke", r"smoke|fire|soot|teflon|ptfe|καπν|φωτια|πυρκαγι", (),
     "Smoke and fire: carbon monoxide, cyanide and soot injure the lungs; swelling of the airways peaks 12 to 24 hours later, so a pet that seems fine still needs a vet. "
     "Overheated non-stick (PTFE) pans can kill birds within minutes.",
     "toxicology/smoke-inhalation-injury/smoke-inhalation-injury-in-animals"),
    ("urine_block", r"urin|pee\b|peeing|litter|\bουρ(α|η|ο)|κατουρ|αμμο", ("dog", "cat"),
     "A blocked urethra (male cats most often; dogs with stones or prostate problems): repeated trips with little or no urine, crying, a hard painful belly, then vomiting and weakness. "
     "Toxins build up in 36 to 48 hours and it can be fatal in about 72, so no urine is an emergency, not constipation.",
     "urinary-system/urolithiasis-in-small-animals/urethral-obstruction-in-small-animals"),
    ("fracture", r"broken|fractur|dislocat|bone|σπασ|κατα[γκ]μα|εξαρθρ", ("dog", "cat"),
     "Suspected fracture or dislocation: keep the pet still and, if needed, carry it on a firm board or blanket; do not try to straighten the limb. "
     "A bone through the skin is an emergency: cover with a clean cloth and go now.",
     "musculoskeletal-system/osteopathies-in-small-animals/bone-fractures-in-dogs-and-cats"),
    ("ticks", r"\btick|ixodes|κροτων|τσιμπουρ", ("dog", "cat"),
     "Tick removal: use fine-tipped tweezers or a tick-removal tool, grasp the tick as close to the skin as possible and pull steadily straight out; do not twist, crush, burn or smother it with oil, alcohol or ether. "
     "Remove visible ticks as soon as possible, because the sooner they are off the less chance of passing on infection. "
     "In Greece the brown dog tick is the usual one; it can pass on ehrlichiosis, babesiosis and hepatozoonosis (Ixodes ticks carry Lyme bacteria across Europe, usually without signs in dogs); "
     "watch for fever, tiredness, lameness, pale gums or poor appetite in the following weeks. Cats get ticks too: never use dog-only permethrin products on a cat. "
     "(Source for the Greek tick and sand-fly points: ESCCAP ectoparasite guideline GL3, esccap.org.)",
     "dog-owners/skin-disorders-of-dogs/ticks-of-dogs"),
    ("med_parasites", r"leishman|sand ?fl(y|ies)|heartworm|dirofil|mosquito|λεισμαν|φλεβοτομ|καρδιοσκωληκ|κουνουπ", ("dog", "cat"),
     "Sand flies and heartworm in Greece (ESCCAP): leishmaniosis is endemic around the Mediterranean and spreads by sand-fly bites, mostly dusk to dawn, "
     "roughly April to November; in Greece protection can be needed all year. Vets use repellent collars or spot-ons on dogs, and permethrin products must never be used on cats. "
     "Signs can be slow: skin sores or scaly hair loss, long nails, swollen glands, weight loss, nosebleeds, drinking and peeing more, lameness, eye problems. "
     "Heartworm is spread by mosquitoes and is endemic in southern Europe: dogs get a chronic cough, tire easily, may faint; cats may cough, breathe fast, vomit, or collapse suddenly. "
     "Prevention is a vet's decision (monthly preventives through the mosquito season, a test first); do not suggest products or doses.",
     "https://www.esccap.org/guidelines/gl5/"),
    ("cecotrophy", r"eat\w* (its|his|her|their) (own )?(poop|droppings|feces|faeces|stool)|cecotrop|night droppings|soft droppings|τρωει τις κουτσουλι|τρωει τα κοπρανα", ("rabbit", "other"),
     "Rabbits, guinea pigs and chinchillas normally eat soft, shiny, nutrient-rich night droppings (cecotropes) straight from the anus; this is healthy and needed. "
     "Worry instead about no droppings, tiny dry droppings, uneaten cecotropes piling up, or diarrhoea.",
     "all-other-pets/rabbits/diet-for-rabbits"),
    ("trauma_watch", r"hit by|fell|fall|kicked|accident|trauma|crush|τροχαι|επεσε|χτυπησε", ("dog", "cat"),
     "After a fall, car or other blunt injury the pet can look fine but hide chest, belly or spine injuries: watch breathing, gum colour and alertness every few hours for the first 48 hours, and see a vet even if the pet seems normal.",
     "emergency-medicine-and-critical-care/specific-diagnostics-and-therapy/trauma-in-emergency-medicine-in-small-animals"),
    ("rat_bait_anticoag", r"rat poison|rodenticide|rat bait|mouse poison|warfarin|brodifacoum|δολωματ|ποντικοφαρμακο|φαρμακο για ποντικ", ("dog", "cat"),
     "Anticoagulant rat bait stops blood clotting, but signs often do not start until more than 24 hours after it was eaten (bruises, nosebleeds, bleeding gums, pale gums, weakness, sudden collapse from internal bleeding). "
     "A pet that looks well after eating rat bait is NOT safe: it needs a vet the same day for treatment before bleeding begins, and the effect can last 2 to 4 weeks with newer products.",
     "toxicology/rodenticide-poisoning/anticoagulant-rodenticide-poisoning"),
    ("drinking_more", r"drinking (much |a lot |far )?more|drinks (a lot|too much)|thirsty|excessive thirst|πινει πολυ|διψα", ("dog", "cat"),
     "A pet that suddenly drinks and pees much more is ill (diabetes, kidney, liver, Cushing's, womb infection and other causes) and needs a vet within a day or two. "
     "Never take the water away or limit it to 'see if it helps': pets that cannot concentrate their urine dehydrate quickly. Always keep fresh water available.",
     "endocrine-system/the-pituitary-gland/diabetes-insipidus-in-animals"),
    ("tortoise_hibernation", r"hibernat|brumat|χειμερια ναρκη|χειμερινος υπνος", ("reptile",),
     "A sick, thin or underweight tortoise or turtle must never be put into hibernation: see a reptile vet first. "
     "Keep a sick reptile in the upper part of its preferred temperature range, give it a proper UVB light (glass and plastic block UVB), and do not force-feed.",
     "all-other-pets/reptiles/providing-a-home-for-a-reptile"),
    ("wound_care", r"bite|bitten|wound|cut\b|gash|puncture|δαγκωμ|πληγ|κοψιμο", ("dog", "cat", "rabbit", "other"),
     "For a bite or wound: press on bleeding with a clean cloth, rinse with plain water or saline, and cover it. Do not use hydrogen peroxide, strong disinfectants or human creams unless a vet says so. "
     "A small bite can hide deep damage (cat bites infect easily, dog bites can break ribs), so every bite should be seen by a vet.",
     "special-pet-topics/emergencies/wound-management"),
    ("anemia_drugs", r"pale gums|yellow (gums|eyes|skin)|jaundice|anemi|anaemi|ικτερ|χλωμα ουλα", ("dog", "cat"),
     "Pale or yellow gums can mean anaemia. Never give human painkillers (paracetamol, aspirin, naproxen) or onion, garlic or fava beans: they can destroy red blood cells. "
     "Tell the vet about recent drugs, vaccines, tick exposure and travel.",
     "dog-owners/blood-disorders-of-dogs/anemia-in-dogs"),
    ("eclampsia", r"nursing|lactat|whelp|gave birth|puppies|kittens|θηλαζ|γεννησε|κουταβια", ("dog", "cat"),
     "Eclampsia (low blood calcium) most often hits small-breed mothers with large litters 1 to 4 weeks after giving birth: restlessness, panting, drooling, stiff walk, twitching, then seizures. "
     "It is an emergency; do not give calcium supplements during pregnancy, as they can make it more likely.",
     "metabolic-disorders/disorders-of-calcium-metabolism/eclampsia-in-small-animals"),
]

from vet_rules import _fold_pat
_COMPILED = [(tid, re.compile(_fold_pat(rx)), sp, txt, (page if page.startswith("http") else MSD + page)) for tid, rx, sp, txt, page in TOPICS]


def relevant(species_key, texts, limit=4):
    """Topics whose keywords appear in the owner's words (normalised), most specific first, at most `limit`."""
    t = norm(" \n ".join(x for x in (texts or []) if x))
    out = []
    for tid, rx, sp, txt, url in _COMPILED:
        if sp and species_key not in sp:
            continue
        if rx.search(t):
            out.append((tid, txt, url))
    return out[:limit]


def prompt_block(species_key, texts, limit=4):
    """Text appended to the nurse system prompt: guardrails plus the matching topic facts. Empty when nothing matches."""
    topics = relevant(species_key, texts, limit)
    if not topics:
        return ""
    lines = [ALWAYS, "REFERENCE FACTS for this conversation (MSD Veterinary Manual and ESCCAP, paraphrased; use them to ask better questions and to give safe first-aid wording, "
             "not to diagnose):"]
    lines += [f"- {txt}" for _tid, txt, _url in topics]
    return "\n".join(lines)
