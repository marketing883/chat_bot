from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import dateparser, phonenumbers
from email_validator import validate_email, EmailNotValidError

import subprocess
from flask import Response
from flask import Flask, request, jsonify, render_template, session
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from pymongo import MongoClient
import secrets
import json
import time
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime, timezone, timedelta
import uuid
from flask_cors import CORS
import glob
from pathlib import Path
import pytz
import re
from pypdf import PdfReader  # parsing helper for Q&A-style PDFs

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_urlsafe(16))

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "./data/")
# Prefer this PDF if present (won't crash if it isn't Q&A-formatted; we now fall back gracefully)
ACI_PDF_NAME = os.getenv("ACI_PDF_NAME", "ACI_RAG_Combined_Text_20250827_0634.txt")
LLM = os.getenv("OPENAI_MODEL", "")  # e.g., "gpt-4o-mini"

# Use 3-large for maximum retrieval quality (3072 dims) unless DB implies otherwise
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
VECTOR_DIM = EMBED_DIMS.get(EMBEDDING_MODEL, 1536)

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Accept JSON, PDF, TXT
ALLOWED_EXTS = (".json", ".pdf", ".txt")

# MongoDB Atlas Search Index configuration
ATLAS_PUBLIC_KEY = os.getenv("ATLAS_PUBLIC_KEY")
ATLAS_PRIVATE_KEY = os.getenv("ATLAS_PRIVATE_KEY")
ATLAS_GROUP_ID = os.getenv("ATLAS_GROUP_ID")
ATLAS_CLUSTER_NAME = os.getenv("ATLAS_CLUSTER_NAME")
DATABASE_NAME = "chat_bot_AI"
INDEX_NAME = "vector_index"
collection_name = "customer_data"

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client.chat_bot_AI
chat_collection = db.chat_history
lead_collection = db.lead_data

# Initialize openai models
llm = ChatOpenAI(
    temperature=LLM_TEMPERATURE,
    openai_api_key=OPENAI_API_KEY,
    model=LLM
)

# Initialize embeddings (default to chosen model)
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model=EMBEDDING_MODEL
)

# Lead extraction model with lower temperature for precision
lead_extraction_llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY,
    model=LLM
)

# Chat history management
chat_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]


user_state_tracker = {}


def get_or_init_user_state(session_id):
    if session_id not in user_state_tracker:
        user_state_tracker[session_id] = {
            "name": None,
            "email_id": None,
            "contact_number": None,
            "job_designation": None,
            "service_interest": None,
            "appointment_date": None,
            "appointment_time": None
        }
    return user_state_tracker[session_id]


# Prompt templates
CONTEXT_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


QA_SYSTEM_PROMPT = """
You are ACIInfotech bot assistant — a warm, helpful virtual guide that speaks on behalf of ACI Infotech, using only the uploaded official documents. You don’t guess, speculate, or use any outside information.

Use the following context from our documents to answer questions:
{context}

I. 🎯 Purpose:
- Provide accurate, helpful answers using ACI Infotech’s uploaded content whenever available.
- If context is missing or unhelpful, use **reasoned, human-like responses** that reflect typical ACI practices (e.g., "Our sessions are typically online via Zoom or Teams").
- Never hallucinate features or services not offered by ACI Infotech.
- Redirect all discussion back to ACI Infotech services and solutions only."

II. 💬 First Greeting:
Show this greeting only if chat_history is empty:
> "Hi, this is ACIInfotech bot assistant! How can I assist you today?"

III. 🗣️ Tone & Style:
- Friendly, approachable, and emotionally intelligent
- Short replies: 2–3 lines max, unless listing services or summarizing content
- Use natural, human tone — avoid robotic phrasing
- No outside content or speculation

IV. 🔄 Lead Capture (strictly sequenced basIntent Guarded on known_lead_data):
Ask **one missing field at a time**, in this exact order:
1. Name → “By the way, may I know your name? It’s always nice to help you personally.”
2. Email → “Could you share your email so we can keep you updated?”
3. Phone → “May I have your phone number in case our team needs to follow up?”
4. Job title → “What’s your job designation?”
5. Service → “Which service or solution are you interested in?”
6. Appointment date → “What date would work best for a short meeting with our expert?”
7. Appointment time → “And what time would you prefer for the call?”

Current lead data: {known_lead_data}
Lead complete: {lead_complete}
User intent: {user_intent}

✅ **Only when `lead_complete` is true**, respond:
> “Thank you, {{known_lead_data[name]}}! You’re all set — your session is confirmed for {{known_lead_data[appointment_date]}} at {{known_lead_data[appointment_time]}}.You’ll receive a meeting link via email shortly. Our team will be in touch soon.”

⚠️ NEVER:
- Ask for a meeting, email, or phone **before name**
- Ask for multiple fields in one message
- Say you have all info unless all 7 fields are truly collected
- Confirm booking unless ALL 7 fields are captured
- Make up dates or times - ALWAYS use the provided current_ist_date and current_ist_time when needed

V. 🧭 History-Anchored Reference Resolution (critical)
Use Chat History to resolve vague or partial service mentions:
- Treat the last assistant bullet list of services you presented as the canonical Service Menu for this session.
- Map ordinals or numbers (first/1st, second/2nd, third/3rd, “option 3”) to that menu by order.
- If the user provides a partial name (“gen ai”), match it to the closest item in that menu.
- If there is no prior menu, propose the closest canonical ACI service name and briefly confirm.
- If ambiguous, present up to 3 likely options and ask the user to choose by name or number.
- Never ask for “company name” — it’s not a required field.

VI. 📅 **DATE AND TIME HANDLING (CRITICAL):**
- When user says "tomorrow" → Use: "tomorrow ({current_ist_date} + 1 day)"
- When user says "today" → Use: "today ({current_ist_date})"
- When user says "same time" → Use: "{current_ist_time}"
- When user asks "what time" or "when" → Always reference the ACTUAL appointment_date and appointment_time from known_lead_data
- NEVER make up dates like "October 4th, 2023" - this is wrong!
- ALWAYS calculate relative to current_ist_date: {current_ist_date}
- ALWAYS use current_ist_time when "same time" is mentioned: {current_ist_time}

VII. 💡 Engagement Hooks (after name is captured):
- “Would you like help choosing the right service?”
- “Would you like a demo or real-world example?”
- “Interested in seeing how other clients benefited?”

 📞 Meeting Booking (only if all 7 fields are captured):
- Confirm the booking like this:
> “Perfect — I’ll arrange the session and our expert will contact you directly.”

VIII. 🔁 Fallback Handling (Improved):
If no document-based answer is found:
- Try to respond helpfully using business context or common sense, while staying within ACI Infotech’s domain.
- If unsure, say:  
  > “That’s a great question — I’ve noted it down and our team will follow up shortly.”
If the user repeats or expresses frustration (e.g., "you said that again"):
- Acknowledge and apologize:  
> “Sorry about that! Let me flag this for our team to get back to you.”
If user gives an invalid date:
- “Feb 30”, “Feb 31”, “Feb 29 (non-leap year)” → reply:
> “That date doesn’t exist. Would March 1st work instead?”
If user keeps entering invalid dates:
- Gently guide:
> “We just need a valid date to schedule your session. Any day next week that works?”
  ---

🚫 Never say:
- “According to the document…”
- “I couldn’t find specific details…”
- Anything speculative or outside ACI content

IX. 🤝 Frustration Handling (NEW SYSTEM BEHAVIOR):
If the user says “I already mentioned,” “you asked again,” or similar:
- Reply warmly: “Thanks for your patience — let me double-check what I have.”
- State the value already stored for that field from known_lead_data (e.g., “I have your service as Generative AI.”).
- Answer their last question.
- Do NOT ask for any new field in this message.

X. 🔍 Link Handling:
Only provide hyperlinks if they appear in the retrieved document context and you are confident they are valid. Do not invent or guess URLs. If no relevant link is present in the documents, describe the service without a link.

XI. Intent Guard (CRITICAL):
- If user_intent == "info" OR the user says “explain first”/“later we’ll schedule”, then:
  • First answer their question clearly.
  • Do NOT ask for any missing lead field in that turn.
  • You may close with a soft CTA like “Happy to schedule a demo whenever you’re ready.”
- If user_intent == "booking" and some fields are missing:
  • Ask exactly ONE missing field next, in the strict order defined above.
  • Do NOT bundle a scheduling question with a field request.
  • Do NOT add any extra questions beyond that single field.

Confirmation:
- Set 'confirm' ONLY if ALL required_fields are present.

When asking, pick the earliest missing field from 'field_order' and set ask_field + ask_prompt.
Never ask for fields already present. Never ask for Company.
Do not confirm unless all required fields are present.

XII. 📝 Message Format:
- Use bullet points for service summaries,but write service names as plain text.
- Keep answers 2–3 lines unless summarizing key offerings


If the user says “I already mentioned / you asked again”, reply:
“Thanks for your patience — let me double-check what I have and answer your question.”
Current IST Date: {current_ist_date}
Current IST Time: {current_ist_time}
Hook Line: {hook_line}
"""

# Create prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXT_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# === Dynamic policy (config-driven) ===
LEAD_POLICY = {
    "required_fields": [
        "name","email_id","contact_number","job_designation",
        "service_interest","appointment_date","appointment_time"
    ],
    "field_order": [
        "name","email_id","contact_number","job_designation",
        "service_interest","appointment_date","appointment_time"
    ],
    "ask_prompts": {
        "name": "By the way, may I know your name? It’s always nice to help you personally.",
        "email_id": "Could you share your email so we can keep you updated?",
        "contact_number": "May I have your phone number in case our team needs to follow up?",
        "job_designation": "What’s your job designation?",
        "service_interest": "Which service or solution are you interested in?",
        "appointment_date": "What date would work best for a short meeting with our expert?",
        "appointment_time": "And what time would you prefer for the call?"
    }
}

class LeadFields(BaseModel):
    name: str = ""
    email_id: str = ""
    contact_number: str = ""
    job_designation: str = ""
    service_interest: str = ""
    appointment_date: str = ""   # raw user text
    appointment_time: str = ""   # raw user text

class Plan(BaseModel):
    intent: Literal["info","booking","mixed"]
    answer_text: str = ""                         # concise info answer (optional)
    extracted: LeadFields = Field(default_factory=LeadFields)
    next_action: Literal["answer_only","ask_next_field","confirm"] = "answer_only"
    ask_field: Optional[str] = None               # chosen from field_order
    ask_prompt: Optional[str] = None              # use ask_prompts[field]
    confirm_text: Optional[str] = None            # only if all fields present

PLANNER_SYSTEM = """
You are a policy engine that decides the next UI turn for an ACI Infotech assistant.
Return STRICT JSON conforming to the provided schema.

Rules:
- Use chat history for context and what’s already collected.
- If the user asks for info/examples, put the concise answer in 'answer_text' (2–3 lines).
- Extract ONLY fields explicitly stated into 'extracted'.

HARD INTENT GUARD:
- If user_intent == "info": you MUST set next_action = "answer_only".
  • Do NOT ask to schedule.
  • Do NOT ask for ANY lead field.
  • You MAY include a soft, non-interrogative CTA (no question mark), e.g., "Happy to schedule a discovery call whenever you’re ready."
- If user_intent == "booking": set next_action = "ask_next_field" unless all fields are present.
  • Ask exactly ONE missing field (earliest in field_order).
  • Do NOT bundle a scheduling question with a field request.
  • Do NOT add any extra questions beyond that single field.

Confirmation:
- Set 'confirm' ONLY if ALL required_fields are present.

When asking, pick the earliest missing field from 'field_order' and set ask_field + ask_prompt.
Never ask for fields already present. Never ask for Company.
Do not confirm unless all required fields are present.
"""

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM),
    ("system", "field_order: {field_order}\nrequired_fields: {required_fields}\nask_prompts: {ask_prompts}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{user_input}")
])

planner = llm.with_structured_output(Plan)
planner_chain = planner_prompt | planner

LEAD_EXTRACTION_PROMPT = """
Extract the following information about the user **ONLY IF the user actually provides it explicitly.** 
Do NOT guess. Only extract:
- name
- email_id
- contact_number
- location
- service_interest
- appointment_date
- appointment_time
- job_designation
Return a raw JSON object with these fields (empty if not provided by the user).
Conversation: {conversation}
"""

# ---------------------------
# Atlas Search Index mgmt
# ---------------------------

def create_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.atlas.2024-05-30+json'}
    data = {
        "collectionName": collection_name,
        "database": DATABASE_NAME,
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": VECTOR_DIM, "similarity": "cosine"}
            ]
        }
    }
    response = requests.post(
        url,
        headers=headers,
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY),
        data=json.dumps(data)
    )
    if response.status_code != 201:
        raise Exception(f"Failed to create Atlas Search Index: {response.status_code}, Response: {response.text}")
    return response


def get_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.get(
        url,
        headers=headers,
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response


def delete_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.delete(
        url,
        headers=headers,
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

# ---- Helpers to keep only the target vector index ----
ATLAS_API_BASE = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes"

def list_collection_indexes():
    """List all Atlas Search/Vector indexes for this collection."""
    url = f"{ATLAS_API_BASE}/{DATABASE_NAME}/{collection_name}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    r = requests.get(url, headers=headers, auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY))
    return r.status_code, (r.json() if r.status_code == 200 else [])

def prune_conflicting_indexes():
    """
    Delete any index on this collection that is NOT our target vector index.
    Keeps only INDEX_NAME when it exists and is type 'vectorSearch'.
    """
    code, items = list_collection_indexes()
    if code != 200:
        print(f"[WARN] Could not list indexes for {DATABASE_NAME}.{collection_name}: HTTP {code}")
        return

    for it in items:
        iname = it.get("name")
        itype = it.get("type")  # 'search' or 'vectorSearch'
        if iname == INDEX_NAME and itype == "vectorSearch":
            continue  # keep our target
        del_url = f"{ATLAS_API_BASE}/{DATABASE_NAME}/{collection_name}/{iname}"
        r = requests.delete(del_url, headers={'Accept': 'application/vnd.atlas.2024-05-30+json'},
                            auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY))
        print(f"[INDEX CLEANUP] delete {iname} ({itype}) -> {r.status_code}")

# ---- NEW HELPERS: choose model/dimension from stored data and create exact-dim index ----
DIM_TO_MODEL = {1536: "text-embedding-3-small", 3072: "text-embedding-3-large"}

def infer_stored_vector_dim():
    """Peek one doc to detect stored vector length; fallback to index definition; else None."""
    sample = db[collection_name].find_one({}, {"embedding": 1})
    if sample and isinstance(sample.get("embedding"), list):
        try:
            return len(sample["embedding"])
        except Exception:
            pass
    try:
        resp = get_atlas_search_index()
        if resp.status_code == 200:
            j = resp.json()
            fields = j.get("definition", {}).get("fields", [])
            for f in fields:
                if f.get("path") == "embedding" and isinstance(f.get("numDimensions"), int):
                    return f["numDimensions"]
    except Exception:
        pass
    return None

def get_embeddings_for_dim(dim: int):
    """Return an OpenAIEmbeddings instance that matches the stored dimension."""
    model = DIM_TO_MODEL.get(dim) or EMBEDDING_MODEL
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=model)

def create_atlas_search_index_for_dim(num_dims: int):
    """Create vector index with explicit numDimensions."""
    if not isinstance(num_dims, int) or num_dims <= 0:
        num_dims = VECTOR_DIM
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.atlas.2024-05-30+json'}
    data = {
        "collectionName": collection_name,
        "database": DATABASE_NAME,
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {"fields": [{"type": "vector", "path": "embedding", "numDimensions": int(num_dims), "similarity": "cosine"}]}
    }
    r = requests.post(url, headers=headers, auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY), data=json.dumps(data))
    if r.status_code != 201:
        raise Exception(f"Failed to create Atlas Search Index: {r.status_code}, Response: {r.text}")
    return r

# ---------------------------
# Unified Document Loading
# ---------------------------

def _docs_from_json_payload(data, source_name):
    docs = []

    def add_records(payload):
        questions = payload.get("question", []) or []
        gts = payload.get("ground_truths", []) or []
        contexts = payload.get("contexts", []) or []
        if contexts and isinstance(contexts, list) and contexts and isinstance(contexts[0], str):
            contexts = [contexts]
        for i, group in enumerate(contexts):
            if not isinstance(group, list):
                continue
            for j, ctx in enumerate(group):
                if not isinstance(ctx, str) or not ctx.strip():
                    continue
                meta = {
                    "source_file": source_name,
                    "file_type": "json",
                    "record_index": i,
                    "chunk_index": j,
                }
                if i < len(questions):
                    meta["question"] = questions[i]
                if i < len(gts):
                    meta["ground_truth"] = gts[i]
                docs.append(Document(page_content=ctx.strip(), metadata=meta))

    if isinstance(data, dict):
        add_records(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                add_records(item)

    return docs

# --- dynamic category heuristics (no hardcoded list) ---

def _is_heading_candidate(line):
    if not line or line.startswith(("Q:", "Sources:")):
        return False
    if any(line.endswith(p) for p in (".", "?", ":", ";")):
        return False
    words = line.split()
    return len(words) <= 6 and len(line) <= 40

def _infer_category_from_urls(urls):
    u = " ".join(urls).lower()
    if "/services/" in u:
        return "Service"
    if "/industry/" in u:
        return "Industry"
    if "/platforms/" in u:
        return "Platform"
    if "/case-studies" in u:
        return "Case Studies"
    if "/blogs" in u or "/insights" in u:
        return "Insights"
    if "/media/" in u or "/press" in u or "/news" in u:
        return "News & Media"
    return "General"

def parse_aci_pdf_records(pdf_path):
    """Parse Q&A style records from a PDF. If parsing fails or no QAs, return []."""
    if not os.path.exists(pdf_path):
        return []
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"[PDF READ WARN] Could not open {pdf_path}: {e}")
        return []

    records = []
    current_heading = None

    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"[PDF PAGE WARN] Failed to extract text on page {page_idx+1}: {e}")
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        i = 0
        while i < len(lines):
            ln = lines[i]
            if _is_heading_candidate(ln):
                current_heading = ln
                i += 1
                continue
            if ln.startswith("Q:"):
                question = ln[2:].strip()
                i += 1
                answer_lines = []
                while i < len(lines) and not lines[i].startswith("Sources:"):
                    answer_lines.append(lines[i])
                    i += 1
                urls = []
                if i < len(lines) and lines[i].startswith("Sources:"):
                    urls = re.findall(r"https?://\S+", lines[i])
                    i += 1
                category = current_heading or _infer_category_from_urls(urls) or "General"
                records.append({
                    "question": question,
                    "answer": "\n".join(answer_lines).strip(),
                    "urls": urls,
                    "category": category,
                    "page": page_idx + 1
                })
            else:
                i += 1
    return records

def load_all_documents():
    """
    Prefer ACI_PDF_NAME if present: attempt Q&A parse; if none or error, fall back to generic loaders.
    Also load any other JSON/PDF/TXT in DOCUMENTS_FOLDER.
    """
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    documents = []

    # 1) Try preferred PDF as Q&A first (no hard failure if none)
    pdf_path = os.path.join(DOCUMENTS_FOLDER, ACI_PDF_NAME)
    if os.path.exists(pdf_path):
        qa_records = parse_aci_pdf_records(pdf_path)
        if qa_records:
            for r in qa_records:
                content = f"Q: {r['question']}\nA:\n{r['answer']}\nSources:\n" + "\n".join(r["urls"])
                meta = {
                    "source_file": ACI_PDF_NAME,
                    "file_type": "pdf",
                    "doc_type": "qa",
                    "question": r["question"],
                    "category": r.get("category", "General"),
                    "urls": r.get("urls", []),
                    "page": r.get("page", None),
                }
                documents.append(Document(page_content=content, metadata=meta))
            print(f"Loaded {len(documents)} Q&A docs from {ACI_PDF_NAME}")
        else:
            print(f"[INFO] No Q&A-style records detected in {ACI_PDF_NAME}; will load as normal PDF.")

    # 2) Load all JSON files
    json_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            jdocs = _docs_from_json_payload(data, os.path.basename(jf))
            if not jdocs:
                print(f"[WARN] No contexts extracted from JSON: {jf}")
            documents.extend(jdocs)
        except Exception as e:
            print(f"[ERROR] Failed parsing JSON {jf}: {e}")

    # 3) Load all PDFs (including the preferred one again if needed)
    pdf_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.pdf"))
    for pf in pdf_files:
        try:
            # Skip re-loading the preferred PDF as generic pages if we already added Q&A docs
            if os.path.basename(pf) == ACI_PDF_NAME and any(d.metadata.get("source_file") == ACI_PDF_NAME and d.metadata.get("doc_type") == "qa" for d in documents):
                continue
            loader = PyPDFLoader(pf)
            pdocs = loader.load()
            for d in pdocs:
                d.metadata["source_file"] = os.path.basename(pf)
                d.metadata["file_type"] = "pdf"
            documents.extend(pdocs)
        except Exception as e:
            print(f"[ERROR] Failed loading PDF {pf}: {e}")

    # 4) Load all TXTs
    txt_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.txt"))
    for tf in txt_files:
        try:
            loader = TextLoader(tf, encoding="utf-8")
            tdocs = loader.load()
            for d in tdocs:
                d.metadata["source_file"] = os.path.basename(tf)
                d.metadata["file_type"] = "txt"
            documents.extend(tdocs)
        except Exception as e:
            print(f"[ERROR] Failed loading TXT {tf}: {e}")

    if not documents:
        raise FileNotFoundError(f"No JSON/PDF/TXT files found in: {DOCUMENTS_FOLDER}")
    print(f"Loaded {len(documents)} docs total")
    return documents


# ---------------------------
# Vector Store Initialization
# ---------------------------

def initialize_vector_store():
    # Load all supported documents (may be PDF Q&A or generic)
    docs = load_all_documents()

    # Do NOT split Q&A docs parsed from the PDF; split others only
    qa_docs = [d for d in docs if d.metadata.get("doc_type") == "qa"]
    other_docs = [d for d in docs if d.metadata.get("doc_type") != "qa"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    final_documents = qa_docs + (text_splitter.split_documents(other_docs) if other_docs else [])

    print(f"Prepared {len(final_documents)} chunks")

    # Remove any existing index FIRST to avoid conflicts when rebuilding from scratch
    response = get_atlas_search_index()
    if response.status_code == 200:
        print("Deleting existing Atlas Search Index...")
        delete_response = delete_atlas_search_index()
        if delete_response.status_code == 204:
            print("Waiting for index deletion to complete...")
            # poll until gone
            for _ in range(30):
                if get_atlas_search_index().status_code == 404:
                    break
                time.sleep(2)
        else:
            raise Exception(f"Failed to delete existing Atlas Search Index: {delete_response.status_code}, Response: {delete_response.text}")
    elif response.status_code != 404:
        raise Exception(f"Failed to check Atlas Search Index: {response.status_code}, Response: {response.text}")

    # Clear existing collection
    db[collection_name].delete_many({})

    # Store embeddings with OpenAI embeddings
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=final_documents,
        embedding=embeddings,
        collection=db[collection_name],
        index_name=INDEX_NAME,
    )

    # Debug: Verify documents in collection
    doc_count = db[collection_name].count_documents({})
    print(f"Number of documents in {collection_name}: {doc_count}")
    if doc_count > 0:
        sample_doc = db[collection_name].find_one()
        print(f"Sample document structure (keys): {list(sample_doc.keys())}")

    # Keep only our vector index
    print("Pruning conflicting Atlas indexes (free-tier limit guard)...")
    prune_conflicting_indexes()

    # Create new Atlas Search Index (vectorSearch) matching our chosen EMBEDDING_MODEL dims
    print("Creating new Atlas Search Index...")
    create_response = create_atlas_search_index()
    print(f"Atlas Search Index creation status: {create_response.status_code}")

    print("Waiting for index to be ready...")
    time.sleep(30)

    return vector_search


# ---------------------------
# Lead extraction helper
# ---------------------------

def update_user_state_from_extraction(session_id, extracted_data):
    current_state = get_or_init_user_state(session_id)
    for field in ["name", "email_id", "contact_number", "job_designation",
                  "service_interest", "appointment_date", "appointment_time"]:
        if extracted_data.get(field) and extracted_data[field].strip():
            current_state[field] = extracted_data[field].strip()
    return current_state

MD_BOLD_RE = re.compile(r"\*\*(.*?)\*\*", flags=re.DOTALL)
MD_UNDER_RE = re.compile(r"__(.*?)__", flags=re.DOTALL)

def sanitize_markdown_to_plain(text: str) -> str:
    if not isinstance(text, str):
        return "" if text is None else str(text)
    text = MD_BOLD_RE.sub(r"\1", text)
    text = MD_UNDER_RE.sub(r"\1", text)
    text = text.replace("**", "")
    return text

# --- Guard: never confirm unless all 7 fields are present ---
CONFIRM_RE = re.compile(
    r"\b(confirm|confirmed|arrange the session|meeting link|we\'ll contact you)\b",
    re.I
)

def block_premature_confirmation(text: str, lead_complete: bool) -> bool:
    """Return True if the text looks like a confirmation but the lead is incomplete."""
    if lead_complete:
        return False
    return bool(CONFIRM_RE.search(text or ""))

def enforce_policy_and_render(plan: Plan, known_lead_data: dict, rag_answer: str, lead_complete: bool, hook_line: str) -> str:
    """
    Compose the final assistant message using the policy:
    - never confirm unless all 7 fields exist
    - if model tried to confirm early, override with the next required ask
    - if plan asks for next field, ask only that field
    - if info turn, answer concisely and (optionally) add soft CTA
    """
    # 1) Hard stop: block any early confirmations from the LLM text
    if block_premature_confirmation(rag_answer, lead_complete):
        missing = next_missing_field(known_lead_data)
        ask = LEAD_POLICY["ask_prompts"].get(missing, "")
        return ask.strip() or "Could you share that next detail?"

    # 2) If planner says confirm, only allow if all fields are truly present
    if plan.next_action == "confirm" or lead_complete:
        if compute_lead_complete(known_lead_data):
            return (
                f"Thank you, {known_lead_data['name']}! You’re all set — your session is confirmed for "
                f"{known_lead_data['appointment_date']} at {known_lead_data['appointment_time']}. "
                f"You’ll receive a meeting link via email shortly. Our team will be in touch soon."
            ).strip()
        # If planner was wrong, fall back to asking the next required field
        missing = next_missing_field(known_lead_data)
        return LEAD_POLICY["ask_prompts"].get(missing, "").strip() or "Could you share that next detail?"

    # 3) If planner wants the next field, do exactly that (no multi-asks)
    if plan.next_action == "ask_next_field" and plan.ask_field:
        return (plan.ask_prompt or LEAD_POLICY["ask_prompts"].get(plan.ask_field, "")).strip() or "Could you share that next detail?"

    # 4) Info / exploration: prefer planner’s concise answer; else RAG answer
    body = (plan.answer_text or rag_answer or "").strip()
    if hook_line and body and hook_line not in body:
        body = f"{body}\n\n{hook_line}"
    return body or "Sure—how can I help further?"

def detect_intent(user_input: str) -> str:
    text = (user_input or "").lower()
    booking_kw = ("book", "schedule", "demo", "meeting", "invite", "call", "appointment")
    info_kw = ("explain", "overview", "details", "tell about", "what is", "services", "info", "brief")
    if any(k in text for k in booking_kw):
        return "booking"
    if any(k in text for k in info_kw):
        return "info"
    return "mixed"

def pick_hook_line(user_intent: str, known_lead_data: dict) -> str:
    name = (known_lead_data.get("name") or "").strip()
    you = f"{name}, " if name else ""
    if user_intent == "info":
        return f"{you}happy to set up a quick discovery call whenever you’re ready."
    if user_intent == "booking":
        return f"{you}I can book it—just share a date and time that works."
    return ""

def extract_lead_info(session_id):
    chat_doc = chat_collection.find_one({"session_id": session_id})
    if not chat_doc or "messages" not in chat_doc:
        return
    recent = chat_doc["messages"][-12:]
    user_only_lines = [
        f"user: {m['content']}"
        for m in recent
        if m.get("role") == "user" and m.get("content")
    ]
    conversation = "\n".join(user_only_lines)

    try:
        response = lead_extraction_llm.invoke(LEAD_EXTRACTION_PROMPT.format(conversation=conversation))
        response_text = response.content.strip()
        if "```json" in response_text or "```" in response_text:
            json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
        try:
            lead_data = json.loads(response_text)
            if lead_data.get("name", "").lower() in ["nisaa", "nisa", "nisaaa"]:
                lead_data["name"] = ""
        except json.JSONDecodeError:
            json_pattern = r'\{[^}]*"name"[^}]*"email_id"[^}]*"contact_number"[^}]*"location"[^}]*"service_interest"[^}]*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            if json_match:
                try:
                    lead_data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    lead_data = {
                        "name": "", "email_id": "", "contact_number": "", "location": "",
                        "service_interest": "", "appointment_date": "", "appointment_time": "",
                        "job_designation": "", "parsing_error": "Failed to parse response"
                    }
            else:
                lead_data = {
                    "name": "", "email_id": "", "contact_number": "", "location": "",
                    "service_interest": "", "appointment_date": "", "appointment_time": "",
                    "job_designation": "", "raw_response": response_text[:500]
                }
        IST = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(IST)
        raw_date = (lead_data.get("appointment_date") or "").strip()
        if raw_date:
            resolved_dt = resolve_relative_date_phrase(raw_date, now_ist)
            if resolved_dt:
                lead_data["appointment_date"] = fmt_ist_date(resolved_dt)
        lead_data["session_id"] = session_id
        lead_data["updated_at"] = datetime.now(timezone.utc)
        lead_data["extraction_model"] = "openai_" + LLM
        lead_collection.update_one(
            {"session_id": session_id},
            {"$set": lead_data},
            upsert=True
        )
    except Exception as e:
        print(f"[Lead Extraction Error] {e}")


# ---------------------------
# Initialize / connect vector store (UPDATED LOGIC)
# ---------------------------

try:
    existing_doc_count = db[collection_name].count_documents({})
    idx_resp = get_atlas_search_index()

    if existing_doc_count > 0:
        # Data already present — DO NOT rebuild or delete. Just ensure an index exists.
        stored_dim = infer_stored_vector_dim()

        if idx_resp.status_code == 404:
            # Create an index that matches the stored vectors' dimension
            create_atlas_search_index_for_dim(stored_dim or VECTOR_DIM)
            time.sleep(10)  # brief wait for index to be ready

        # Use embeddings that match the stored vector dimension for querying
        effective_dim = stored_dim or VECTOR_DIM
        embeddings = get_embeddings_for_dim(effective_dim)

        vector_search = MongoDBAtlasVectorSearch(
            collection=db[collection_name],
            embedding=embeddings,
            index_name=INDEX_NAME,
        )
        print(f"Using existing vector store with {existing_doc_count} documents and dim {effective_dim}")

    else:
        # No data yet — build from documents (embeds + creates index)
        print("No embedded documents found — initializing vector store...")
        vector_search = initialize_vector_store()

except Exception as e:
    print(f"Failed to initialize or connect to vector store: {e}")
    raise

# --- Pull safe links from retrieved doc text (ACI domain only; avoid guessed URLs) ---
LINK_RE = re.compile(r"https?://[^\s)>\]]+")
ACI_DOMAINS = ("aciinfotech.com", "www.aciinfotech.com")
DENY_SUBPATHS = ("/pla/",)

def extract_links_from_docs(docs):
    urls = []
    seen = set()
    for d in docs or []:
        for u in (d.metadata.get("urls") or []):
            u2 = u.rstrip(".,);]\"")
            if any(dom in u2 for dom in ACI_DOMAINS) and not any(bad in u2 for bad in DENY_SUBPATHS):
                if u2 not in seen:
                    urls.append(u2); seen.add(u2)
    if not urls:
        for d in docs or []:
            text = (getattr(d, "page_content", "") or "")
            for m in LINK_RE.findall(text):
                u = m.rstrip(".,);]\"")
                if any(dom in u for dom in ACI_DOMAINS) and not any(bad in u for bad in DENY_SUBPATHS):
                    if u not in seen:
                        urls.append(u); seen.add(u)
    return urls[:3]

def compute_lead_complete(known_lead_data: dict) -> bool:
    required = [
        "name", "email_id", "contact_number",
        "job_designation", "service_interest",
        "appointment_date", "appointment_time",
    ]
    return all(((known_lead_data.get(k) or "")).strip() for k in required)

def normalize_lead_fields(extracted: dict, now_ist):
    out = dict(extracted or {})

    # email
    if out.get("email_id"):
        try:
            out["email_id"] = validate_email(out["email_id"], check_deliverability=False).normalized
        except EmailNotValidError:
            out["email_id"] = ""

    # phone
    if out.get("contact_number"):
        try:
            pn = phonenumbers.parse(out["contact_number"], "IN")
            if phonenumbers.is_valid_number(pn):
                out["contact_number"] = phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164)
            else:
                out["contact_number"] = ""
        except Exception:
            out["contact_number"] = ""

    # date (free-form phrases like "in 2 days", "next Tue", "26/08")
    if out.get("appointment_date"):
        dt = dateparser.parse(
            out["appointment_date"],
            settings={
                "TIMEZONE": "Asia/Kolkata",
                "TO_TIMEZONE": "Asia/Kolkata",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "RELATIVE_BASE": now_ist
            }
        )
        out["appointment_date"] = dt.strftime("%B %d, %Y") if dt else ""

    # time (support "same time", "5 pm", "17:30")
    if out.get("appointment_time"):
        ttext = (out["appointment_time"] or "").lower()
        if "same time" in ttext:
            out["appointment_time"] = now_ist.strftime("%I:%M %p IST")
        else:
            tt = dateparser.parse(
                out["appointment_time"],
                settings={
                    "TIMEZONE": "Asia/Kolkata",
                    "TO_TIMEZONE": "Asia/Kolkata",
                    "RETURN_AS_TIMEZONE_AWARE": True,
                    "RELATIVE_BASE": now_ist
                }
            )
            out["appointment_time"] = tt.strftime("%I:%M %p IST") if tt else ""

    return out

def next_missing_field(known: dict):
    for f in LEAD_POLICY["field_order"]:
        if not (known.get(f) or "").strip():
            return f
    return None


# --- Relative date resolution (IST) ---
WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}

def _next_weekday(from_dt, target_weekday_idx: int) -> datetime:
    days_ahead = (target_weekday_idx - from_dt.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return from_dt + timedelta(days=days_ahead)

def resolve_relative_date_phrase(raw_text: str, now_ist: datetime):
    if not raw_text:
        return None
    text = raw_text.strip().lower()
    if text == "today":
        return now_ist
    if text.startswith("tomorrow"):
        return now_ist + timedelta(days=1)
    m = re.search(r"\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text)
    if m:
        when, day = m.group(1), m.group(2)
        target_idx = WEEKDAY_MAP[day]
        if when == "next":
            return _next_weekday(now_ist, target_idx)
        else:
            days_ahead = (target_idx - now_ist.weekday() + 7) % 7
            return now_ist + timedelta(days=days_ahead)
    m2 = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text)
    if m2:
        return _next_weekday(now_ist, WEEKDAY_MAP[m2.group(1)])
    return None

def fmt_ist_date(d: datetime) -> str:
    return d.strftime("%B %d, %Y")


# ---------------------------
# Chat handler
# ---------------------------

def handle_chat(session_id: str, user_input: str) -> str:
    chat_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "messages": {
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now(timezone.utc)
                }
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True
    )
    # Keep server truth in sync with any user-provided fields in recent turns
    extract_lead_info(session_id)

    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retriever = vector_search.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3}
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    IST = pytz.timezone('Asia/Kolkata')
    current_ist = datetime.now(IST)
    lead_doc = lead_collection.find_one({"session_id": session_id}) or {}
    known_lead_data = {
        "name": lead_doc.get("name", ""),
        "email_id": lead_doc.get("email_id", ""),
        "contact_number": lead_doc.get("contact_number", ""),
        "job_designation": lead_doc.get("job_designation", ""),
        "service_interest": lead_doc.get("service_interest", ""),
        "appointment_date": lead_doc.get("appointment_date", ""),
        "appointment_time": lead_doc.get("appointment_time", ""),
    }
    # Coerce None/non-strings to safe strings to avoid .strip() crashes downstream
    for _k, _v in list(known_lead_data.items()):
        if _v is None:
            known_lead_data[_k] = ""
        elif not isinstance(_v, str):
            known_lead_data[_k] = str(_v)

    # === 1) Planner decides intent + next action dynamically
    plan: Plan = planner_chain.invoke({
        "field_order": LEAD_POLICY["field_order"],
        "required_fields": LEAD_POLICY["required_fields"],
        "ask_prompts": LEAD_POLICY["ask_prompts"],
        "chat_history": get_session_history(session_id).messages,
        "user_input": user_input
    })
    # Guardrail: never ask for surname/company—stick to required fields only
    if plan.ask_field and plan.ask_field not in LEAD_POLICY["field_order"]:
        plan.ask_field = next_missing_field(known_lead_data)
        plan.ask_prompt = LEAD_POLICY["ask_prompts"].get(plan.ask_field, plan.ask_prompt)

    # === 2) Normalize whatever fields the planner extracted this turn
    now_ist = current_ist
    normalized = normalize_lead_fields(plan.extracted.model_dump(), now_ist)

    # === 3) Persist any newly extracted fields (server truth)
    to_set = {}
    for k, v in normalized.items():
        if not v:
            continue

        # Guard: never accept junk names and don't overwrite an existing name casually
        if k == "name":
            invalid_names = {"anyone", "any one", "someone", "no one", "none"}
            proposed = v.strip().lower()
            if proposed in invalid_names:
                continue
            # Only allow changing name if the user clearly states it
            if known_lead_data.get("name"):
                if not re.search(r'\b(my name is|this is|i am|call me)\b', (user_input or "").lower()):
                    continue

        to_set[k] = v

    if to_set:
        lead_collection.update_one(
            {"session_id": session_id},
            {"$set": {**to_set, "session_id": session_id, "updated_at": datetime.now(timezone.utc)}},
            upsert=True
        )
        known_lead_data.update(to_set)

    # === 4) Determine completion
    lead_complete = compute_lead_complete(known_lead_data)

    # === 5) Build the response
    user_intent = detect_intent(user_input)
    hook_line = pick_hook_line(user_intent, known_lead_data) or ""

    response = conversational_rag_chain.invoke(
        {
            "input": user_input,
            "known_lead_data": known_lead_data,
            "lead_complete": lead_complete,
            "user_intent": user_intent,
            "hook_line": hook_line,
            "current_ist_date": current_ist.strftime("%B %d, %Y"),
            "current_ist_time": current_ist.strftime("%I:%M %p IST")
        },
        config={"configurable": {"session_id": session_id}}
    )

    answer = sanitize_markdown_to_plain(response.get("answer", ""))

    docs = response.get("context", [])
    want_links = any(k in (user_input or "").lower() for k in ("service", "services", "ai", "case study", "case studies", "industry"))
    if want_links:
        links = extract_links_from_docs(docs)
        if links:
            answer += "\n\nResources:\n" + "\n".join(f"- {u}" for u in links)

    chat_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "messages": {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now(timezone.utc)
                }
            }
        },
        upsert=True
    )
    return answer


# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/check_ffmpeg")
def check_ffmpeg():
    try:
        import platform
        ffmpeg_check_cmd = "where ffmpeg" if platform.system() == "Windows" else "which ffmpeg"
        output = subprocess.check_output(ffmpeg_check_cmd, shell=True, stderr=subprocess.STDOUT)
        return Response(output, mimetype="text/plain")
    except Exception as e:
        return f"FFmpeg not found or failed to execute: {e}", 500


@app.route('/generate_session', methods=['GET'])
def generate_session():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    session_id = data.get('session_id', str(uuid.uuid4()))

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
        answer = handle_chat(session_id, user_input)
        return jsonify({'response': answer}), 200
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/leads', methods=['GET'])
def get_leads():
    leads = list(lead_collection.find({}, {"_id": 0}))
    return jsonify(leads)


@app.route('/upload_documents', methods=['POST'])
def upload_documents():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        uploaded_files = []

        for file in files:
            if file.filename == '':
                continue
            if not file.filename.lower().endswith(ALLOWED_EXTS):
                continue
            filename = file.filename
            file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
            file.save(file_path)
            uploaded_files.append(filename)

        if uploaded_files:
            global vector_search
            # Freshly (re)index uploaded docs
            vector_search = initialize_vector_store()
            return jsonify({
                'message': f'Successfully uploaded and processed {len(uploaded_files)} file(s)',
                'files': uploaded_files
            }), 200
        else:
            return jsonify({'error': 'No valid JSON/PDF/TXT files uploaded'}), 400

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    try:
        client.admin.command('ping')
        doc_count = db[collection_name].count_documents({})
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'documents_indexed': doc_count,
            'model': LLM,
            'embedding_model': EMBEDDING_MODEL
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == "__main__":
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=8000, debug=True)
