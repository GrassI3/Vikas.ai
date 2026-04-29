"""
generate_full_dataset.py — Comprehensive Instruction-Tuning Dataset Generator

Bakes ALL knowledge into training examples:
  1. Every condition from graph.jsonld × every language
  2. Every safety rule from safety_rules.json × variant phrasings
  3. Multi-symptom combination scenarios
  4. Edge cases: romanized, code-switched, Devanagari, native scripts

Output: full_training_data.jsonl (2000+ examples)
"""

import json
import os
import random
import itertools
from copy import deepcopy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

KG_PATH = os.path.join("knowledge_graph", "graph.jsonld")
SAFETY_PATH = os.path.join("knowledge_graph", "safety_rules.json")
OUTPUT_FILE = "full_training_data.jsonl"

# ---------------------------------------------------------------------------
# All Indian Languages — translations for the dataset
# ---------------------------------------------------------------------------

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
}

SEVERITY_LABELS = {
    "LOW":       {"en": "Low",       "hi": "कम",      "mr": "कमी",     "bn": "কম",       "te": "తక్కువ",   "ta": "குறைவு",    "gu": "ઓછું",     "kn": "ಕಡಿಮೆ",    "ml": "കുറവ്",     "pa": "ਘੱਟ"},
    "MODERATE":  {"en": "Moderate",  "hi": "मध्यम",    "mr": "मध्यम",   "bn": "মাঝারি",   "te": "మితమైన",   "ta": "மிதமான",    "gu": "મધ્યમ",    "kn": "ಮಧ್ಯಮ",    "ml": "മിതമായ",   "pa": "ਦਰਮਿਆਨਾ"},
    "HIGH":      {"en": "High",      "hi": "गंभीर",    "mr": "गंभीर",   "bn": "উচ্চ",     "te": "అధిక",     "ta": "உயர்",      "gu": "ઉચ્ચ",     "kn": "ಹೆಚ್ಚು",     "ml": "ഉയർന്ന",    "pa": "ਉੱਚ"},
    "EMERGENCY": {"en": "Emergency", "hi": "आपातकालीन", "mr": "आणीबाणी", "bn": "জরুরী",    "te": "అత్యవసర",  "ta": "அவசரம்",    "gu": "કટોકટી",   "kn": "ತುರ್ತು",    "ml": "അടിയന്തരം", "pa": "ਐਮਰਜੈਂਸੀ"},
}

SEVERITY_DESCRIPTIONS = {
    "LOW": {
        "en": "This appears to be a mild condition. Monitor symptoms at home.",
        "hi": "यह एक हल्की स्थिति प्रतीत होती है। घर पर लक्षणों की निगरानी करें।",
        "mr": "ही सौम्य स्थिती असल्याचे दिसते. घरी लक्षणांवर लक्ष ठेवा.",
        "bn": "এটি একটি মৃদু অবস্থা বলে মনে হচ্ছে। বাড়িতে লক্ষণগুলি পর্যবেক্ষণ করুন।",
        "te": "ఇది తేలికపాటి పరిస్థితిగా కనిపిస్తోంది. ఇంట్లో లక్షణాలను గమనించండి.",
        "ta": "இது ஒரு லேசான நிலையாகத் தெரிகிறது. வீட்டிலேயே அறிகுறிகளைக் கண்காணிக்கவும்.",
        "gu": "આ એક હળવી સ્થિતિ હોવાનું જણાય છે. ઘરે લક્ષણોનું નિરીક્ષણ કરો.",
        "kn": "ಇದು ಸೌಮ್ಯ ಸ್ಥಿತಿ ಎಂದು ತೋರುತ್ತದೆ. ಮನೆಯಲ್ಲಿ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
        "ml": "ഇതൊരു ചെറിയ അവസ്ഥയാണെന്ന് തോന്നുന്നു. വീട്ടിൽ രോഗലക്ഷണങ്ങൾ നിരീക്ഷിക്കുക.",
        "pa": "ਇਹ ਇੱਕ ਹਲਕੀ ਸਥਿਤੀ ਜਾਪਦੀ ਹੈ। ਘਰ ਵਿੱਚ ਲੱਛਣਾਂ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ।"
    },
    "MODERATE": {
        "en": "This needs medical attention. Please consult a doctor within 24 hours.",
        "hi": "इसे चिकित्सा ध्यान की आवश्यकता है। कृपया 24 घंटे के भीतर डॉक्टर से परामर्श करें।",
        "mr": "यासाठी वैद्यकीय लक्ष आवश्यक आहे. कृपया 24 तासांच्या आत डॉक्टरांचा सल्ला घ्या.",
        "bn": "এর জন্য চিকিৎসার প্রয়োজন। দয়া করে 24 ঘন্টার মধ্যে একজন ডাক্তারের পরামর্শ নিন।",
        "te": "దీనికి వైద్య సహాయం అవసరం. దయచేసి 24 గంటలలోపు వైద్యుడిని సంప్రదించండి.",
        "ta": "இதற்கு மருத்துவ கவனம் தேவை. 24 மணி நேரத்திற்குள் மருத்துவரை அணுகவும்.",
        "gu": "આ માટે તબીબી ધ્યાનની જરૂર છે. કૃપા કરીને 24 કલાકની અંદર ડૉક્ટરની સલાહ લો.",
        "kn": "ಇದಕ್ಕೆ ವೈದ್ಯಕೀಯ ಗಮನ ಅಗತ್ಯ. ದಯವಿಟ್ಟು 24 ಗಂಟೆಗಳ ಒಳಗೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "ml": "ഇതിന് വൈദ്യസഹായം ആവശ്യമാണ്. 24 മണിക്കൂറിനുള്ളിൽ ഒരു ഡോക്ടറെ സമീപിക്കുക.",
        "pa": "ਇਸ ਵੱਲ ਡਾਕਟਰੀ ਧਿਆਨ ਦੇਣ ਦੀ ਲੋੜ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ 24 ਘੰਟਿਆਂ ਦੇ ਅੰਦਰ ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਲਓ।"
    },
    "HIGH": {
        "en": "This is a serious condition. Seek medical care urgently.",
        "hi": "यह एक गंभीर स्थिति है। तुरंत चिकित्सा सहायता लें।",
        "mr": "ही गंभीर स्थिती आहे. तातडीने वैद्यकीय मदत घ्या.",
        "bn": "এটি একটি গুরুতর অবস্থা। জরুরি ভিত্তিতে চিকিৎসা নিন।",
        "te": "ఇది తీవ్రమైన పరిస్థితి. అత్యవసరంగా వైద్య సంరక్షణ పొందండి.",
        "ta": "இது ஒரு தீவிரமான நிலை. அவசரமாக மருத்துவ உதவியை நாடுங்கள்.",
        "gu": "આ એક ગંભીર સ્થિતિ છે. તાત્કાલિક તબીબી સંભાળ લો.",
        "kn": "ಇದು ಗಂಭೀರ ಸ್ಥಿತಿಯಾಗಿದೆ. ತುರ್ತಾಗಿ ವೈದ್ಯಕೀಯ ಆರೈಕೆಯನ್ನು ಪಡೆಯಿರಿ.",
        "ml": "ഇതൊരു ഗുരുതരമായ അവസ്ഥയാണ്. അടിയന്തരമായി വൈദ്യസഹായം തേടുക.",
        "pa": "ਇਹ ਇੱਕ ਗੰਭੀਰ ਸਥਿਤੀ ਹੈ। ਤੁਰੰਤ ਡਾਕਟਰੀ ਦੇਖਭਾਲ ਲਓ।"
    },
    "EMERGENCY": {
        "en": "EMERGENCY — Seek immediate medical attention or call 108.",
        "hi": "आपातकाल — तुरंत चिकित्सा सहायता लें या 108 पर कॉल करें।",
        "mr": "आणीबाणी — तातडीने वैद्यकीय मदत घ्या किंवा 108 वर कॉल करा.",
        "bn": "জরুরী — অবিলম্বে চিকিৎসা সহায়তা নিন বা 108 নম্বরে কল করুন।",
        "te": "అత్యవసర — తక్షణ వైద్య సహాయం తీసుకోండి లేదా 108కి కాల్ చేయండి.",
        "ta": "அவசரம் — உடனடியாக மருத்துவ உதவியை நாடுங்கள் அல்லது 108 ஐ அழைக்கவும்.",
        "gu": "કટોકટી — તાત્કાલિક તબીબી ધ્યાન લો અથવા 108 પર કૉલ કરો.",
        "kn": "ತುರ್ತು — ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ನೆರವು ಪಡೆಯಿರಿ ಅಥವಾ 108 ಗೆ ಕರೆ ಮಾಡಿ.",
        "ml": "അടിയന്തരം — ഉടനടി വൈദ്യസഹായം തേടുക അല്ലെങ്കിൽ 108 ൽ വിളിക്കുക.",
        "pa": "ਐਮਰਜੈਂਸੀ — ਤੁਰੰਤ ਡਾਕਟਰੀ ਸਹਾਇਤਾ ਲਓ ਜਾਂ 108 'ਤੇ ਕਾਲ ਕਰੋ।"
    }
}

ADVICE = {
    "LOW": [
        {"en": "Rest and stay hydrated", "hi": "आराम करें और पानी पीते रहें"},
        {"en": "Monitor symptoms for 24-48 hours", "hi": "24-48 घंटे लक्षणों पर नज़र रखें"},
        {"en": "Visit a doctor if symptoms worsen", "hi": "लक्षण बिगड़ने पर डॉक्टर को दिखाएं"},
    ],
    "MODERATE": [
        {"en": "Consult a doctor within 24 hours", "hi": "24 घंटे के भीतर डॉक्टर से मिलें"},
        {"en": "Avoid strenuous activity", "hi": "भारी शारीरिक गतिविधि से बचें"},
    ],
    "HIGH": [
        {"en": "Seek urgent medical care — visit ER or call your doctor now", "hi": "तुरंत ER जाएं या डॉक्टर को कॉल करें"},
        {"en": "Do not ignore worsening symptoms", "hi": "बिगड़ते लक्षणों को नज़रअंदाज़ न करें"},
    ],
    "EMERGENCY": [
        {"en": "Call 108 (emergency ambulance) IMMEDIATELY", "hi": "तुरंत 108 (एम्बुलेंस) पर कॉल करें"},
        {"en": "If available, go to the nearest Emergency Room", "hi": "निकटतम आपातकालीन कक्ष जाएं"},
    ],
}

SYSTEM_PROMPT = (
    "You are Sahaayak, an AI medical triage assistant for India. "
    "Analyze the patient's symptoms and respond with a JSON object containing: "
    "severity (LOW/MODERATE/HIGH/EMERGENCY), severity_label (in detected language), "
    "severity_description (in detected language), symptoms_detected (list), "
    "possible_causes (list with condition name, severity, escalation_triggers), "
    "related_conditions (list), recommended_actions (list with priority and action text in detected language), "
    "and safety_flags (list of triggered safety alerts with exact action text). "
    "Always respond in the language the patient uses. "
    "For emergencies, always include the number 108 and relevant crisis helplines."
)

# ---------------------------------------------------------------------------
# Load knowledge sources
# ---------------------------------------------------------------------------

def load_kg():
    with open(KG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_safety_rules():
    with open(SAFETY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def build_output(severity: str, lang: str, symptoms: list, causes: list,
                 related: list, actions: list, safety_flags: list) -> dict:
    """Build the target JSON output the model should learn to generate."""
    return {
        "language_detected": lang,
        "severity": severity,
        "severity_label": SEVERITY_LABELS.get(severity, {}).get(lang, severity),
        "severity_description": SEVERITY_DESCRIPTIONS.get(severity, {}).get(lang, ""),
        "symptoms_detected": symptoms,
        "possible_causes": causes,
        "related_conditions": related,
        "recommended_actions": actions,
        "safety_flags": safety_flags,
    }


def generate_single_condition_examples(kg_data: dict) -> list:
    """Generate examples for each condition × each idiom."""
    examples = []

    for node in kg_data.get("@graph", []):
        clinical_name = node.get("med:clinicalName", "")
        severity = node.get("med:defaultSeverity", "LOW")
        related = [r.replace("med:", "") for r in node.get("med:relatedSymptoms", [])]
        escalation = node.get("med:escalationTriggers", [])

        for idiom in node.get("indic:idioms", []):
            term = idiom.get("indic:term", "")
            lang = idiom.get("indic:lang", "en")
            literal = idiom.get("indic:literal", "")
            context = idiom.get("indic:context", "")

            # Romanized input prompt
            prompt_variants = [
                f"mujhe {term} ho raha hai",
                f"mere papa ko {term} ki problem hai",
                f"{term} bahut zyada ho raha hai",
                f"I am experiencing {term}",
                f"{term} since morning",
            ]

            symptoms = [{
                "condition": clinical_name,
                "idiom_matched": term,
                "literal": literal,
                "context": context,
                "default_severity": severity,
            }]

            causes = [{
                "condition": clinical_name,
                "severity": severity,
                "severity_label": SEVERITY_LABELS.get(severity, {}).get(lang, severity),
                "escalation_triggers": escalation,
            }]

            actions = [
                {"priority": severity, "action": a.get(lang, a.get("en", ""))}
                for a in ADVICE.get(severity, [])
            ]

            output = build_output(severity, lang, symptoms, causes, related, actions, [])

            for prompt in prompt_variants:
                examples.append({
                    "system": SYSTEM_PROMPT,
                    "input": prompt,
                    "output": json.dumps(output, ensure_ascii=False),
                })

    return examples


def generate_safety_rule_examples(safety_data: dict) -> list:
    """Generate examples for every safety rule × variant phrasings."""
    examples = []

    for rule in safety_data.get("rules", []):
        flag = rule["flag"]
        action = rule["action"]
        override = rule["override_to"]

        for keyword in rule["keywords"]:
            # Create various phrasings
            prompts = [
                f"mujhe {keyword} ho raha hai please help",
                f"my father is experiencing {keyword}",
                f"{keyword} bahut serious hai kya karu",
                f"patient has {keyword} symptoms",
                f"emergency {keyword} help needed",
            ]

            safety_flags = [{
                "flag": flag,
                "action": action,
                "matched_keyword": keyword,
            }]

            actions = [
                {"priority": "CRITICAL", "action": action, "flag": flag},
                {"priority": "EMERGENCY", "action": "Call 108 (emergency ambulance) IMMEDIATELY"},
                {"priority": "EMERGENCY", "action": "If available, go to the nearest Emergency Room"},
            ]

            # Add crisis helplines for mental health
            if flag == "CRISIS_MENTAL_HEALTH":
                actions.append({"priority": "CRITICAL", "action": "iCall: 9152987821. Vandrevala Foundation: 1860-2662-345."})

            for lang_code in ["en", "hi"]:
                output = build_output("EMERGENCY", lang_code, [], [], [], actions, safety_flags)

                for prompt in prompts:
                    examples.append({
                        "system": SYSTEM_PROMPT,
                        "input": prompt,
                        "output": json.dumps(output, ensure_ascii=False),
                    })

    return examples


def generate_multi_symptom_examples(kg_data: dict) -> list:
    """Generate examples where multiple conditions appear together."""
    examples = []
    nodes = kg_data.get("@graph", [])

    # Create pairs of conditions
    for i in range(len(nodes)):
        for j in range(i + 1, min(i + 4, len(nodes))):
            node_a = nodes[i]
            node_b = nodes[j]

            idioms_a = node_a.get("indic:idioms", [])
            idioms_b = node_b.get("indic:idioms", [])

            if not idioms_a or not idioms_b:
                continue

            idiom_a = random.choice(idioms_a)
            idiom_b = random.choice(idioms_b)

            term_a = idiom_a["indic:term"]
            term_b = idiom_b["indic:term"]

            prompt = f"mujhe {term_a} bhi hai aur {term_b} bhi ho raha hai"

            sev_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "EMERGENCY": 3}
            sev_a = node_a.get("med:defaultSeverity", "LOW")
            sev_b = node_b.get("med:defaultSeverity", "LOW")
            final_sev = sev_a if sev_rank.get(sev_a, 0) >= sev_rank.get(sev_b, 0) else sev_b

            symptoms = [
                {
                    "condition": node_a.get("med:clinicalName", ""),
                    "idiom_matched": term_a,
                    "literal": idiom_a.get("indic:literal", ""),
                    "context": idiom_a.get("indic:context", ""),
                    "default_severity": sev_a,
                },
                {
                    "condition": node_b.get("med:clinicalName", ""),
                    "idiom_matched": term_b,
                    "literal": idiom_b.get("indic:literal", ""),
                    "context": idiom_b.get("indic:context", ""),
                    "default_severity": sev_b,
                },
            ]

            causes = [
                {"condition": node_a.get("med:clinicalName", ""), "severity": sev_a,
                 "severity_label": SEVERITY_LABELS.get(sev_a, {}).get("hi", sev_a),
                 "escalation_triggers": node_a.get("med:escalationTriggers", [])},
                {"condition": node_b.get("med:clinicalName", ""), "severity": sev_b,
                 "severity_label": SEVERITY_LABELS.get(sev_b, {}).get("hi", sev_b),
                 "escalation_triggers": node_b.get("med:escalationTriggers", [])},
            ]

            related_a = [r.replace("med:", "") for r in node_a.get("med:relatedSymptoms", [])]
            related_b = [r.replace("med:", "") for r in node_b.get("med:relatedSymptoms", [])]
            related = list(set(related_a + related_b))

            actions = [
                {"priority": final_sev, "action": a.get("hi", a.get("en", ""))}
                for a in ADVICE.get(final_sev, [])
            ]

            output = build_output(final_sev, "hi", symptoms, causes, related, actions, [])

            examples.append({
                "system": SYSTEM_PROMPT,
                "input": prompt,
                "output": json.dumps(output, ensure_ascii=False),
            })

    return examples


def generate_english_clinical_examples() -> list:
    """Generate pure English medical phrasing examples."""
    cases = [
        ("I have a severe headache and nausea since morning", "LOW", "Cephalgia"),
        ("Patient presenting with high fever, chills, and body aches for 3 days", "LOW", "Pyrexia"),
        ("Experiencing chest pain radiating to left arm with shortness of breath", "EMERGENCY", "Acute Chest Pain"),
        ("Feeling very dizzy, room is spinning, and I almost fainted", "MODERATE", "Vertigo / Presyncope"),
        ("Severe abdominal pain in the lower right side with nausea", "MODERATE", "Abdominal Pain"),
        ("Difficulty breathing at rest, lips turning blue", "EMERGENCY", "Dyspnea"),
        ("Persistent loose stools and vomiting for 2 days, feeling dehydrated", "LOW", "Acute Diarrheal Disease"),
        ("Sudden numbness on one side of body, slurred speech", "EMERGENCY", "Cerebrovascular Accident"),
        ("Feeling hopeless, no will to live, thinking about ending it all", "EMERGENCY", "Suicidal Ideation — CRISIS"),
        ("Found my grandfather unconscious on the floor, not responding", "EMERGENCY", "Loss of Consciousness"),
    ]

    examples = []
    for text, severity, condition in cases:
        symptoms = [{"condition": condition, "idiom_matched": "", "literal": "", "context": text, "default_severity": severity}]
        causes = [{"condition": condition, "severity": severity, "severity_label": severity.capitalize(), "escalation_triggers": []}]
        actions = [{"priority": severity, "action": a.get("en", "")} for a in ADVICE.get(severity, [])]

        safety_flags = []
        if severity == "EMERGENCY" and "Suicidal" in condition:
            safety_flags = [{"flag": "CRISIS_MENTAL_HEALTH", "action": "Crisis intervention. iCall: 9152987821. Vandrevala: 1860-2662-345."}]
        elif severity == "EMERGENCY" and "Chest" in condition:
            safety_flags = [{"flag": "CARDIAC_ALERT", "action": "Recommend calling 108 immediately."}]
        elif severity == "EMERGENCY" and "Cerebrovascular" in condition:
            safety_flags = [{"flag": "STROKE_ALERT", "action": "FAST protocol. Call 108 immediately."}]

        output = build_output(severity, "en", symptoms, causes, [], actions, safety_flags)
        examples.append({
            "system": SYSTEM_PROMPT,
            "input": text,
            "output": json.dumps(output, ensure_ascii=False),
        })

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Sahaayak Full Dataset Generator")
    print("=" * 60)

    kg = load_kg()
    safety = load_safety_rules()

    print("\n[1/4] Generating single-condition examples (KG idioms)...")
    single = generate_single_condition_examples(kg)
    print(f"       -> {len(single)} examples")

    print("[2/4] Generating safety rule examples...")
    safety_ex = generate_safety_rule_examples(safety)
    print(f"       -> {len(safety_ex)} examples")

    print("[3/4] Generating multi-symptom combination examples...")
    multi = generate_multi_symptom_examples(kg)
    print(f"       -> {len(multi)} examples")

    print("[4/4] Generating English clinical examples...")
    english = generate_english_clinical_examples()
    print(f"       -> {len(english)} examples")

    all_examples = single + safety_ex + multi + english
    random.shuffle(all_examples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[SUCCESS] Done! Wrote {len(all_examples)} examples to {OUTPUT_FILE}")
    print(f"   File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # Print a sample
    sample = random.choice(all_examples)
    print(f"\n--- Sample ---")
    print(f"Input: {sample['input']}")
    print(f"Output: {json.dumps(json.loads(sample['output']), indent=2, ensure_ascii=False)[:500]}...")


if __name__ == "__main__":
    main()
