import streamlit as st
import nltk
import re
import math
import io
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pypdf import PdfReader

st.set_page_config(page_title="NyayaLens - Legal NLP", layout="wide")

# download nltk packages on first run
@st.cache_resource
def load_nltk():
    packages = [
        'punkt', 'punkt_tab', 'stopwords', 'wordnet',
        'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'omw-1.4'
    ]
    for p in packages:
        nltk.download(p, quiet=True)

load_nltk()

# sample judgments covering Supreme Court, High Court, and PIL formats
SAMPLES = {
    "State of Karnataka vs. Ramesh Kumar (Supreme Court Criminal Appeal)": """IN THE SUPREME COURT OF INDIA
CRIMINAL APPELLATE JURISDICTION
CRIMINAL APPEAL NO. 1428 OF 2023

STATE OF KARNATAKA ...APPELLANT(S)
VERSUS
RAMESH KUMAR & ORS. ...RESPONDENT(S)

CORAM: HON'BLE JUSTICE D.Y. CHANDRACHUD, HON'BLE JUSTICE J.B. PARDIWALA
Date of Judgment: 14th August 2023

J U D G M E N T
1. This criminal appeal arises out of the impugned judgment dated 12th January 2022 passed by the High Court of Karnataka at Bengaluru. The High Court set aside the conviction of the respondent under Section 302 of the Indian Penal Code, 1860.
2. Briefly stated, the facts of the case are that on 5th May 2019, the accused Ramesh Kumar attacked the victim Suresh with an iron rod at the Mysuru Market following a sudden quarrel over a property dispute. The victim sustained severe head injuries and was admitted to Apollo Hospital, where he succumbed to his injuries two days later.
3. The Police Inspector detained the suspect at the scene of the crime and registered an FIR No. 112/2019 under Section 302 and Section 324 of the Indian Penal Code. The Trial Court convicted the accused and sentenced him to life imprisonment.
4. The short question for consideration before this Court is whether the offence committed by the accused amounts to murder under Section 302 IPC or culpable homicide not amounting to murder under Section 304 Part II IPC.
5. Learned counsel for the appellant argued that the medical evidence clearly establishes that the bodily injury was sufficient in the ordinary course of nature to cause death. He submitted that the High Court erred in acquitting the respondent.
6. Conversely, learned counsel for the respondent contended that the incident occurred without premeditation in a sudden fight in the heat of passion. Therefore, the counsel argued that the offence falls under Exception 4 to Section 300 of the IPC and is punishable only under Section 304 Part II of the Indian Penal Code.
7. In the landmark precedent of Virsa Singh vs. State of Punjab (AIR 1958 SC 465), this Court held that the prosecution must prove that the bodily injury was intended and was not accidental. Similarly, in Pulicherla Nagaraju vs. State of A.P. (2006) 11 SCC 444, the Court laid down parameters to distinguish murder from culpable homicide not amounting to murder.
8. Having examined the witness testimonies and the post-mortem report, we find that there was no prior enmity between the appellant and the respondent. The weapon was picked up from the spot during a heated verbal altercation. Thus, the ratio decidendi is that where a fatal blow is inflicted without premeditation during a sudden quarrel, the culpable intention under Section 302 IPC cannot be attributed to the accused.
9. In view of the above discussion, the appeal is partly allowed. We alter the conviction of the respondent from Section 302 IPC to Section 304 Part II of the Indian Penal Code and sentence him to rigorous imprisonment for 7 years. The order of the High Court is modified accordingly.""",

    "Verma Enterprises vs. National Bank (Delhi High Court - Cyber Fraud)": """IN THE HIGH COURT OF DELHI AT NEW DELHI
CIVIL WRIT JURISDICTION
W.P.(C) NO. 8892 OF 2024

M/S VERMA ENTERPRISES PVT. LTD. .....Petitioner
Through: Mr. Rajeev Mehra, Adv.
versus
NATIONAL COMMERCIAL BANK & ANR. .....Respondents
Through: Ms. Sunita Sharma, Adv.

CORAM:
HON'BLE MS. JUSTICE PRATHIBA M. SINGH
Date of Judgment: 22nd March 2024

JUDGMENT
1. The petitioner filed this writ petition under Article 226 of the Constitution of India seeking compensation for unauthorized fraudulent transactions amounting to Rs. 18,50,000 from their corporate current account.
2. The factual matrix reveals that on 10th November 2023, unknown cyber fraudsters bypassed the two-factor authentication system of the respondent bank and transferred funds from the petitioner's account at the Connaught Place branch. The petitioner immediately notified the Branch Manager within two hours of receiving the SMS alert.
3. Learned counsel for the petitioner submitted that under Section 43A of the Information Technology Act, 2000 and the Reserve Bank of India Circular dated 6th July 2017, a customer has zero liability when an unauthorized electronic banking transaction occurs due to a third-party breach where neither the bank nor the customer is at fault, provided the customer notifies the bank within three working days.
4. Learned counsel for the respondent bank contended that the petitioner was negligent in safeguarding their corporate login credentials. He argued that the dispute involves complex questions of fact and should be adjudicated by a Civil Court rather than under writ jurisdiction.
5. This Court relied on the precedent established in State Bank of India vs. Shiby Thomas (2021), wherein it was held that banks are statutory custodians of public funds and must maintain robust cybersecurity infrastructure.
6. Upon perusal of the forensic audit report, it is evident that no OTP was shared by the petitioner, and the security breach occurred due to a vulnerability in the bank's API gateway. The legal principle is clear that a bank cannot shift the burden of systemic technical vulnerabilities onto an innocent depositor who reported the fraud promptly.
7. Consequently, the writ petition is allowed. The respondent bank is directed to refund the entire sum of Rs. 18,50,000 to the petitioner's account along with interest at 6% per annum within four weeks from today.""",

    "Citizens Forum vs. Union of India (Supreme Court Environment PIL)": """2024 INSC 782
IN THE SUPREME COURT OF INDIA
ORIGINAL CIVIL JURISDICTION
WRIT PETITION (CIVIL) NO. 512 OF 2024

CITIZENS FOR CLEAN ENVIRONMENT FORUM ...PETITIONER(S)
Versus
UNION OF INDIA & STATE POLLUTION CONTROL BOARD ...RESPONDENT(S)

J U D G M E N T
1. The petitioner organization instituted this Public Interest Litigation under Article 32 of the Constitution of India complaining against severe untreated industrial effluents being discharged by chemical factories into the Yamuna River.
2. The facts on record demonstrate that despite multiple show-cause notices issued by the State Pollution Control Board under Section 33A of the Water (Prevention and Control of Pollution) Act, 1974, the industrial units continued to release hazardous waste without functional effluent treatment plants.
3. Learned counsel for the petitioner submitted that the right to a pollution-free environment and clean drinking water is an integral facet of the Fundamental Right to Life guaranteed under Article 21 of the Constitution of India.
4. Learned counsel appearing for the State Pollution Control Board contended that periodic inspections were conducted and penalties were imposed on defaulting factories, and requested six months to implement a common treatment plant.
5. In Vellore Citizens Welfare Forum vs. Union of India (AIR 1996 SC 2715) and M.C. Mehta vs. Union of India (1987), this Court categorically established that the 'Polluter Pays Principle' and the 'Precautionary Principle' are essential features of sustainable development in Indian environmental jurisprudence.
6. We are of the considered view that administrative delay cannot be a ground to permit the continuous violation of citizens' fundamental rights under Article 21. The ratio decidendi of our environmental jurisprudence mandates strict and immediate liability against industries that degrade public water bodies.
7. Accordingly, the writ petition is disposed of with strict directions. The State Pollution Control Board shall immediately seal all non-compliant industrial units within 15 days and recover environmental compensation of Rs. 50 Lakhs from the defaulting respondents.
........................J.
[B.R. GAVAI]
........................J.
[K.V. VISWANATHAN]
NEW DELHI;
OCTOBER 05, 2024"""
}


# 1. Read file and clean universal court formatting noise
def read_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    else:
        return file.read().decode("utf-8", errors="ignore")


def clean_court_noise(text):
    # remove Supreme Court / High Court digital signature blocks and watermarks
    text = re.sub(r"Digitally signed by.*?(?:IST|Reason\s*:?)", " ", text, flags=re.I | re.S)
    text = re.sub(r"Signing Date\s*:\s*[\d\.\-\s:]+", " ", text, flags=re.I)
    text = re.sub(r"I attest to the accuracy and\s*authenticity of this order/judgment", " ", text, flags=re.I)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text, flags=re.I)

    # normalize spaced out headers like "J U D G M E N T" or "O R D E R"
    text = re.sub(r"\bJ\s+U\s+D\s+G\s+M\s+E\s+N\s+T\b", "JUDGMENT.", text)
    text = re.sub(r"\bO\s+R\s+D\s+E\s+R\b", "ORDER.", text)
    text = re.sub(r"\bP\s+R\s+A\s+Y\s+E\s+R\b", "PRAYER.", text)

    # fix hyphenated line breaks and collapse single newlines
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def preprocess_text(text):
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 3]
    clean_txt = clean_court_noise(text)

    raw_sents = sent_tokenize(clean_txt)
    # filter out short header fragments, standalone numbers, and signature dots (< 5 words)
    sents = []
    for s in raw_sents:
        s_clean = s.strip()
        # skip lines that are mostly dots or header titles
        if len(s_clean.split()) >= 5 and not re.search(r"(?:\.{4,}|…{2,})", s_clean):
            sents.append(s_clean)

    tokens = word_tokenize(clean_txt)
    words = [w for w in tokens if w.isalpha() and len(w) > 2]

    sw = set(stopwords.words("english"))
    sw.update({
        "page", "insc", "reportable", "versus", "ors", "anr", "shri", "smt",
        "learned", "hon", "ble", "also", "said", "would", "may", "shall", "upon"
    })
    filtered = [w for w in words if w.lower() not in sw]

    ps = PorterStemmer()
    wnl = WordNetLemmatizer()

    morph_data = []
    seen = set()
    for w in filtered:
        lw = w.lower()
        if lw not in seen and len(lw) > 4:
            seen.add(lw)
            stem_w = ps.stem(lw)
            lemma_w = wnl.lemmatize(lw, pos='v')
            if lemma_w == lw:
                lemma_w = wnl.lemmatize(lw, pos='n')

            morph_data.append({
                "Word": w,
                "Stemmed (Porter)": stem_w,
                "Lemmatized (WordNet)": lemma_w
            })

    return paras, sents, tokens, words, filtered, pd.DataFrame(morph_data)


# 2. Universal NER across SC, HC, eCourts & Indian Kanoon
def get_legal_entities(text, sents):
    header_block = text[:4500]
    footer_block = text[-2500:]
    flat_text = re.sub(r"\s+", " ", text)

    # A. Petitioner / Appellant / Complainant / Plaintiff
    pet_list = []
    for m in re.findall(r"^\s*(?:Petitioner|Appellant|Complainant|Plaintiff|Applicant)s?\s*:\s*([^\n]+)", header_block, re.I | re.M):
        pet_list.append(m)
    for m in re.findall(r"([^\n]+?)\s*(?:\.{2,}|…+)\s*(?:PETITIONER|APPELLANT|COMPLAINANT|PLAINTIFF|APPLICANT)", header_block, re.I):
        pet_list.append(m)
    for m in re.findall(r"^\s*([A-Z][A-Za-z0-9\.\s&\(\)]+?)\s+(?:vs\.?|v\.?)\s+[A-Z][A-Za-z0-9\.\s&\(\)]+?\s+on\s+\d", header_block, re.I | re.M):
        pet_list.append(m)

    # B. Respondent / Accused / Defendant
    resp_list = []
    for m in re.findall(r"^\s*(?:Respondent|Accused|Defendant|Opposite Party)s?\s*:\s*([^\n]+)", header_block, re.I | re.M):
        resp_list.append(m)
    for m in re.findall(r"([^\n]+?)\s*(?:\.{2,}|…+)\s*(?:RESPONDENT|ACCUSED|DEFENDANT|OPPOSITE PARTY)", header_block, re.I):
        resp_list.append(m)
    for m in re.findall(r"^\s*[A-Z][A-Za-z0-9\.\s&\(\)]+?\s+(?:vs\.?|v\.?)\s+([A-Z][A-Za-z0-9\.\s&\(\)]+?)\s+on\s+\d", header_block, re.I | re.M):
        resp_list.append(m)

    # C. Judges / Coram / Bench (Header & Footer)
    judges = []
    for m in re.findall(r"^\s*(?:Coram|Bench|Author)\s*:\s*([^\n]+)", header_block, re.I | re.M):
        judges.append(m)
    for m in re.findall(r"HON'BLE\s+(?:MR\.|MS\.|MRS\.|DR\.|SHRI|SMT\.)?\s*JUSTICE\s+([A-Z][A-Za-z\.\s]+?)(?=\n|HON'BLE|JUDGMENT|ORDER|,|and|$)", header_block, re.I):
        judges.append("Justice " + m.strip().title())
    for m in re.findall(r"(?:\.{2,}|…+)\s*J\.\s*\n+\s*[\[\(]([A-Z][A-Z\.\s]+?)[\]\)]", footer_block):
        judges.append("Justice " + m.strip().title())
    for m in re.findall(r"\n\s*([A-Z][A-Z\.\s]{3,35}?),\s*J\.\s*\n", header_block):
        judges.append("Justice " + m.strip().title())

    # D. Courts
    court_pats = r"(?:Supreme Court of India|High Court of Judicature at [A-Za-z]+|High Court of [A-Za-z\s]+?(?= at | in |\n|,|\.)|High Court|District\s*(?:and|&)\s*Sessions Court|Family Court|Trial Court|Civil Court)"
    courts = [c.strip().title() for c in re.findall(court_pats, flat_text[:3000], re.I)]

    # E. Case Numbers / Neutral Citations / CNR
    case_nos = re.findall(
        r"(?:\b\d{4}\s+INSC\s+\d+\b|(?:CRIMINAL|CIVIL)\s+APPEAL\s+NO\.?\s*\d+\s+OF\s+\d{4}|(?:WRIT\s+PETITION|W\.P\.|SLP|SPECIAL\s+LEAVE\s+PETITION|CRM-M|RC\.REV)\s*\(?[A-Za-z]*\)?\s*NO\.?\s*[\d/\-]+\s*(?:OF\s+\d{4})?|CNR\s+No\.?\s*[A-Z0-9]+|FIR\s+No\.?\s*[\d/]+)",
        flat_text[:3000], re.I
    )

    # F. Sections / Articles / Statutes (Supports IPC, BNS, BNSS, CrPC, CPC, Constitution, Special Acts)
    secs = re.findall(
        r"(?:Section\s+\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z\-]+\))?(?:\s+Part\s+[I|V|X]+)?|Article\s+\d+[A-Z]*(?:\(\d+\))?(?:\([a-z]\))?|Order\s+[IVX]+\s+Rule\s+\d+|Exception\s+\d+|Indian Penal Code|Bharatiya Nyaya Sanhita|Bharatiya Nagarik Suraksha Sanhita|Code of Criminal Procedure|Code of Civil Procedure|Information Technology Act|Negotiable Instruments Act|Right of Children to Free and Compulsory Education Act|Constitution of India)",
        flat_text, re.I
    )

    # G. Dates
    dates = re.findall(
        r"(?:\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{2}[\./\-]\d{2}[\./\-]\d{4}\b)",
        flat_text, re.I
    )

    # H. Precedents & Reports Cited
    cases = re.findall(r"[A-Z][a-zA-Z\.\s&]+?\s+(?:vs\.?|v\.?)\s+[A-Z][a-zA-Z\.\s&\(\)]+?(?=\s*\(|\s*,|\s*\d{4}|\s*AIR|\s*SCC|\s*\n)", text)
    reports = re.findall(r"(?:[A-Z][a-zA-Z]+\s+Commission\s+Report|[A-Z][a-zA-Z]+\s+Committee)", flat_text)

    # I. Monetary Amounts
    money = re.findall(r"(?:Rs\.?|INR|₹)\s*\d[\d,]*(?:\.\d+)?(?:\s*Lakhs?|\s*Crores?|\s*Thousand)?", flat_text, re.I)

    clean_pet = list(dict.fromkeys([re.sub(r"\s+", " ", p).strip(" .,:-_…") for p in pet_list if 2 < len(p.strip()) < 85]))
    clean_resp = list(dict.fromkeys([re.sub(r"\s+", " ", r).strip(" .,:-_…") for r in resp_list if 2 < len(r.strip()) < 85]))
    clean_judges = list(dict.fromkeys([re.sub(r"\s+", " ", j).strip(" .,:-_[]()") for j in judges if 3 < len(j.strip()) < 70]))
    clean_cases = list(dict.fromkeys([re.sub(r"\s+", " ", c).strip(" .,:-_") for c in (cases + reports) if 5 < len(c.strip()) < 80]))

    info = {
        "Petitioner / Appellant": clean_pet[:2] if clean_pet else ["Mentioned in body text"],
        "Respondent / Accused": clean_resp[:2] if clean_resp else ["Mentioned in body text"],
        "Judges / Coram": clean_judges[:4],
        "Courts": list(dict.fromkeys(courts))[:3],
        "Case / Citation Nos": list(dict.fromkeys([c.strip() for c in case_nos]))[:4],
        "Sections & Acts": list(dict.fromkeys([s.strip() for s in secs]))[:15],
        "Dates": list(dict.fromkeys([d.strip() for d in dates]))[:6],
        "Precedents Cited": clean_cases[:8],
        "Amounts": list(dict.fromkeys([m.strip() for m in money]))[:6]
    }

    pos_list = []
    for s in sents[:5]:
        tagged = nltk.pos_tag(word_tokenize(s))
        for word, tag in tagged[:10]:
            pos_list.append({"Word": word, "POS Tag": tag})

    return info, pd.DataFrame(pos_list)


def get_srl_roles(sents):
    rows = []
    helper_verbs = {'is', 'was', 'were', 'are', 'be', 'been', 'have', 'had', 'has', 'do', 'does', 'did'}

    for s in sents:
        tagged = nltk.pos_tag(word_tokenize(s))

        verb_idx = -1
        action_verb = ""
        for i, (w, t) in enumerate(tagged):
            if t.startswith("VB") and w.lower() not in helper_verbs and len(w) > 2:
                verb_idx = i
                action_verb = w
                break

        if verb_idx == -1:
            continue

        left_part = tagged[:verb_idx]
        right_part = tagged[verb_idx+1:]

        agent_tokens = [w for w, t in left_part if t.startswith("NN") or t.startswith("JJ") or t == "PRP"]
        agent = " ".join(agent_tokens[-4:]) if agent_tokens else "Court / Authority"

        right_text = " ".join([w for w, _ in right_part])

        inst = "-"
        m_inst = re.search(r"\b(?:with|using|by|under|vide)\s+([a-zA-Z0-9\s\(\)\-\.]+?)(?=\b(?:at|in|on|from|to|and|for)\b|[,\.]|$)", right_text, re.I)
        if m_inst:
            inst = m_inst.group(0).strip()[:45]

        loc = "-"
        m_loc = re.search(r"\b(?:at|in|into|from|before)\s+([a-zA-Z0-9\s]+?)(?=\b(?:with|under|following|within|and|on)\b|[,\.]|$)", right_text, re.I)
        if m_loc:
            loc = m_loc.group(0).strip()[:45]

        theme_tokens = [w for w, t in right_part[:7] if t.startswith("NN") or t.startswith("JJ") or t == "CD"]
        theme = " ".join(theme_tokens[:4]) if theme_tokens else right_text[:35]

        rows.append({
            "Agent (Who)": agent,
            "Action (Verb)": action_verb,
            "Target / Theme": theme,
            "Instrument / Law": inst,
            "Location / Forum": loc
        })

    return pd.DataFrame(rows)


def resolve_pronouns(sents):
    target_pronouns = {"he", "him", "his", "she", "her", "they", "their"}
    skip_words = {
        'section', 'article', 'court', 'india', 'act', 'ipc', 'page',
        'state', 'high', 'supreme', 'order', 'judgment', 'clause'
    }

    data = []
    current_entity = "Petitioner / Appellant"

    for i, s in enumerate(sents):
        tokens = word_tokenize(s)
        tagged = nltk.pos_tag(tokens)

        proper = [w for w, t in tagged if t == "NNP" and w.lower() not in skip_words and len(w) > 2]
        found = [w.lower() for w in tokens if w.lower() in target_pronouns]

        for p in set(found):
            data.append({
                "Sent No": i + 1,
                "Pronoun": p,
                "Refers To": current_entity,
                "Sentence": s
            })

        if proper:
            current_entity = " ".join(proper[:3])

    return pd.DataFrame(data)


# 3. Universal Rhetorical Roles (Aligned with Indian Kanoon & Legal-NLP)
def label_rhetorical_roles(sents):
    data = []
    for i, s in enumerate(sents):
        low = s.lower()
        if any(w in low for w in [
            "appeal is allowed", "partly allowed", "disposed of", "directed to refund",
            "alter the conviction", "writ petition is allowed", "seal all", "modified accordingly",
            "we direct", "we further direct", "list this special leave petition", "shall be impleaded",
            "copy of our order shall be sent", "stands dismissed", "is hereby quashed",
            "petition is allowed", "appeal fails", "ordered accordingly", "bail is granted"
        ]):
            role = "Final Verdict / Order"
            bg = "#d4edda"
        elif any(w in low for w in [
            "vs.", "v.", "precedent", "this court held", "relied on", "air 19", "scc online",
            "kothari commission", "tendulkar committee", "sustaining section 12", "coordinate bench",
            "reliance placed by", "reported as"
        ]):
            role = "Precedent Cited"
            bg = "#fff3cd"
        elif any(w in low for w in [
            "question for consideration", "issue for consideration", "whether the",
            "point for determination", "short question", "issue under consideration"
        ]):
            role = "Issue / Question of Law"
            bg = "#fce5cd"
        elif any(w in low for w in [
            "ratio decidendi", "considered view", "legal principle", "we find that",
            "upon perusal", "having examined", "we are of the opinion", "we have held",
            "it is important to recognize", "it is necessary to underscore", "our concern is",
            "constitutional jurisprudence compels", "in conclusion, we hold", "we agree with",
            "extraordinary capacity to transform", "i have heard learned counsel"
        ]):
            role = "Ratio Decidendi (Reasoning)"
            bg = "#d1ecf1"
        elif any(w in low for w in [
            "argued that", "contended that", "submitted that", "learned counsel",
            "amicus curiae", "he would submit", "joint suggestions", "highlighted by",
            "additional solicitor general", "counsel appearing for", "pleaded that", "contends that"
        ]):
            role = "Counsel Arguments"
            bg = "#f8d7da"
        elif any(w in low for w in [
            "facts of the case", "factual matrix", "briefly stated", "attacked", "transferred funds",
            "detained", "registered an fir", "show-cause", "petitioner had approached",
            "it is his case that", "compelled to approach", "turned back", "zila parishad",
            "seats are still lying vacant", "marriage between", "solemnized on", "alleges that"
        ]):
            role = "Case Facts"
            bg = "#e2e3e5"
        elif any(w in low for w in [
            "duty bearer", "standard operating procedure", "sop", "preparatory stage",
            "window for correction", "dispute settlement", "reimbursement", "section 12",
            "article 21a", "shall provide free and compulsory", "sub-section", "clause ("
        ]):
            role = "Statutory Provision / Law"
            bg = "#e8daef"
        else:
            role = "Case Background"
            bg = "#f8f9fa"

        data.append({
            "No": i + 1,
            "Role": role,
            "Sentence": s,
            "Color": bg
        })
    return pd.DataFrame(data)


def summarize_tfidf(sents, role_df, ratio=0.30):
    sw = set(stopwords.words("english"))
    sw.update({
        "page", "insc", "reportable", "versus", "ors", "anr", "section", "act",
        "court", "high", "supreme", "state", "india", "learned", "shall", "also"
    })
    n_sents = len(sents)
    if n_sents == 0:
        return [], role_df, pd.DataFrame()

    sent_words = []
    doc_freq = Counter()

    for s in sents:
        clean_w = [w.lower() for w in word_tokenize(s) if w.isalpha() and len(w) > 2 and w.lower() not in sw]
        sent_words.append(clean_w)
        for uw in set(clean_w):
            doc_freq[uw] += 1

    idf = {}
    for w, df in doc_freq.items():
        idf[w] = math.log(n_sents / (df + 1)) + 1.0

    scores = []
    tfidf_table = []

    for i, words in enumerate(sent_words):
        if len(words) < 4:
            scores.append(0.0)
            continue

        tf_counts = Counter(words)
        total = len(words)
        sent_score = 0.0

        for w, count in tf_counts.items():
            tf_val = count / total
            idf_val = idf[w]
            tfidf_val = tf_val * idf_val
            sent_score += tfidf_val

            tfidf_table.append({
                "Word": w,
                "TF": round(tf_val, 4),
                "IDF": round(idf_val, 4),
                "TF-IDF": round(tfidf_val, 4)
            })

        # smooth length normalization so short headers don't dominate
        norm_score = sent_score / math.sqrt(total + 5.0)

        # boost score for core legal discourse roles
        r = role_df.iloc[i]["Role"]
        if r in ["Final Verdict / Order", "Ratio Decidendi (Reasoning)"]:
            norm_score = norm_score * 1.45
        elif r in ["Issue / Question of Law", "Case Facts", "Counsel Arguments", "Precedent Cited"]:
            norm_score = norm_score * 1.25

        scores.append(round(norm_score, 4))

    role_df = role_df.copy()
    role_df["Score"] = scores

    k = max(3, int(n_sents * ratio))
    top_idx = sorted(role_df.nlargest(k, "Score").index.tolist())
    summary = [role_df.iloc[idx] for idx in top_idx]

    top_words_df = pd.DataFrame(tfidf_table).groupby("Word", as_index=False).agg({
    "TF": "mean", "IDF": "first", "TF-IDF": "sum"
}).round(4)
    top_words_df = top_words_df.sort_values(by="TF-IDF", ascending=False).head(15)

    return summary, role_df, top_words_df


# --- Main App UI ---
st.title("NyayaLens: Court Judgment Analysis using NLP")
st.write("NLP Course Project - Pre-processing, Entity Extraction, Semantic Role Labeling & Summarization")

st.sidebar.header("Judgment Input")
mode = st.sidebar.radio("Select Source:", ["Sample Judgments", "Upload File (PDF/TXT)"])

text_input = ""
if mode == "Sample Judgments":
    choice = st.sidebar.selectbox("Choose Case:", list(SAMPLES.keys()))
    text_input = SAMPLES[choice]
else:
    up_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if up_file is not None:
        text_input = read_file(up_file)

st.sidebar.markdown("---")
comp_ratio = st.sidebar.slider("Summary Length (%)", 15, 70, 30, 5) / 100.0

if not text_input.strip():
    st.warning("Please select a sample case or upload a judgment file from the sidebar.")
    st.stop()

# Run functions
paras, sents, tokens, words, filtered, morph_df = preprocess_text(text_input)
entities, pos_df = get_legal_entities(text_input, sents)
srl_df = get_srl_roles(sents)
coref_df = resolve_pronouns(sents)
role_df = label_rhetorical_roles(sents)
summary_list, scored_df, top_words_df = summarize_tfidf(sents, role_df, comp_ratio)

# Summary stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Paragraphs", len(paras))
col2.metric("Clean Sentences", len(sents))
col3.metric("Total Tokens", len(tokens))
col4.metric("Keywords (No Stopwords)", len(filtered))

with st.expander("Show Full Judgment Text"):
    st.text_area("Raw Text", text_input, height=200)

tab1, tab2, tab3 = st.tabs([
    "1. Text Pre-Processing",
    "2. NER & Semantic Roles",
    "3. Discourse & Summary"
])

with tab1:
    st.subheader("Tokenization, Stopwords & Lemmatization")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Sentence Tokenization (Noise & Footers Removed)**")
        sent_df = pd.DataFrame({"No": range(1, len(sents)+1), "Sentence": sents})
        st.dataframe(sent_df, use_container_width=True, height=250)

    with c2:
        st.write("**Top 15 Frequent Words (After Stopword Removal)**")
        freq = Counter([w.lower() for w in filtered]).most_common(15)
        f_df = pd.DataFrame(freq, columns=["Word", "Count"])
        st.bar_chart(f_df.set_index("Word"))

    st.write("**Stemming vs Lemmatization Comparison**")
    st.dataframe(morph_df, use_container_width=True, height=260)

with tab2:
    st.subheader("Named Entity Recognition (NER) & Semantic Role Labeling")

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.write("**Petitioner / Appellant:**", ", ".join(entities["Petitioner / Appellant"]))
        st.write("**Respondent / Accused:**", ", ".join(entities["Respondent / Accused"]))
        st.write("**Judges / Coram:**", ", ".join(entities["Judges / Coram"]) or "-")
    with ec2:
        st.write("**Courts:**", ", ".join(entities["Courts"]) or "-")
        st.write("**Case / Citation Nos:**", ", ".join(entities["Case / Citation Nos"]) or "-")
        st.write("**Dates:**", ", ".join(entities["Dates"]) or "-")
    with ec3:
        st.write("**Acts & Sections:**", ", ".join(entities["Sections & Acts"]) or "-")
        st.write("**Precedents Cited:**", ", ".join(entities["Precedents Cited"]) or "-")
        st.write("**Amounts:**", ", ".join(entities["Amounts"]) or "-")

    st.markdown("---")
    st.write("**Semantic Role Labeling (Who did What to Whom)**")
    st.dataframe(srl_df, use_container_width=True)

    st.markdown("---")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.write("**Pronoun / Anaphora Resolution**")
        st.dataframe(coref_df, use_container_width=True, height=220)
    with sc2:
        st.write("**POS Tagging Sample**")
        st.dataframe(pos_df, use_container_width=True, height=220)

with tab3:
    st.subheader("Rhetorical Role Classification & TF-IDF Extractive Summary")

    role_counts = scored_df["Role"].value_counts().to_dict()
    st.write("**Detected Discourse Structure:**", " | ".join([f"{k}: {v}" for k, v in role_counts.items()]))

    st.write(f"**Generated Summary ({len(summary_list)} of {len(sents)} sentences):**")
    for item in summary_list:
        st.markdown(f"- **[{item['Role']}]** {item['Sentence']}")

    st.markdown("---")
    st.write("**Sentence-wise Rhetorical Roles**")
    for _, row in scored_df.iterrows():
        st.markdown(
            f"<div style='background-color:{row['Color']}; padding:8px; border-radius:4px; margin-bottom:6px; color:#000;'>"
            f"<b>#{row['No']} [{row['Role']}] (Score: {row['Score']})</b>: {row['Sentence']}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.write("**Top TF-IDF Weighted Words**")
    st.dataframe(top_words_df, use_container_width=True)
