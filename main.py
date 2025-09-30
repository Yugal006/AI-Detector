import re
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import logging
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from deep_translator import GoogleTranslator
import language_tool_python
import torch
import math
import uvicorn
from collections import Counter
from sentence_transformers import SentenceTransformer, util


warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used when initializing.*")


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AITextDetector")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


# Enhanced code detection function using detailed regex patterns
def looks_like_code(text):
    code_patterns = [
        # Python keywords and function/class definitions
        r"\b(def|class|import|return|if|else|elif|while|for|try|except|lambda|with|as|from|print|yield|async|await)\b",
        # Common code punctuation
        r"[{}();\[\]]",
        # Single line comments (C, C++, Java, JS)
        r"//.*?$",
        # Python style comments
        r"#.*?$",
        # JavaScript keywords and function arrow syntax
        r"\b(function|var|let|const|=>|console\.log)\b",
        # Multi-line string delimiters for Python/JS
        r"\"\"\"|'''",
        # HTML/XML tags (start and end)
        r"<\/?[a-zA-Z]+\s*[^>]*>",
        # Bash variables or command substitution
        r"\$[a-zA-Z_]\w*",
        r"`[^`]+`",  # backtick command substitution
        # TypeScript/Java function typing
        r":\s*(string|number|boolean|any|void|Promise|Array)<.*?>",
        # Common keywords for Java, C#
        r"\b(public|private|protected|static|final|new|this|super)\b",
        # Regexes often found in code / literals
        r"/.+?/[gimuy]*",
        # Numeric literals typical in code
        r"\b\d+\b",
    ]
    combined_pattern = "|".join(code_patterns)
    return bool(re.search(combined_pattern, text, flags=re.MULTILINE))


# Load models
MODEL_DETECTOR_NAME = "roberta-base-openai-detector"
tokenizer_detector = None
model_detector = None
try:
    tokenizer_detector = AutoTokenizer.from_pretrained(MODEL_DETECTOR_NAME)
    model_detector = AutoModelForSequenceClassification.from_pretrained(MODEL_DETECTOR_NAME)
    model_detector.eval()
    logger.info(f"Loaded model: {MODEL_DETECTOR_NAME}")
except Exception as e:
    logger.error(f"Failed to load model {MODEL_DETECTOR_NAME} - {e}")

GPT2_NAME = "gpt2"
tokenizer_gpt2 = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
model_gpt2 = GPT2LMHeadModel.from_pretrained(GPT2_NAME)
model_gpt2.eval()

tool = language_tool_python.LanguageTool('en-US')

# Load Sentence Transformer for semantic similarity (embedding)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast

# Reference human-like pure text embedding (can be refined)
REFERENCE_HUMAN_EMBEDDING = sentence_model.encode("This is a typical human written text.", convert_to_tensor=True)


def calculate_perplexity(text: str) -> float:
    encodings = tokenizer_gpt2(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_gpt2(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())


def perplexity_to_percentage(perplexity: float) -> int:
    pct = min(max((perplexity - 10) / 40 * 100, 0), 100)
    return round(pct)


def detect_repeated_lines(lines):
    counter = Counter([line.strip() for line in lines if line.strip()])
    repeated = {line: count for line, count in counter.items() if count > 1}
    return repeated


def grammar_mistakes(text):
    matches = tool.check(text)
    return len(matches)


def semantic_similarity_score(text):
    """Calculate cosine similarity of input text embedding to reference human embedding."""
    emb = sentence_model.encode(text, convert_to_tensor=True)
    cos_sim = util.cos_sim(emb, REFERENCE_HUMAN_EMBEDDING).item()
    # Invert similarity for AI likelihood (lower similarity means more AI-like)
    ai_score = (1 - cos_sim) * 100
    return min(max(ai_score, 0), 100)


def combine_scores(roberta_score, perplexity_score, semantic_score, grammar_errors, repeated_lines_count):
    """
    Combine multiple heuristics and model outputs into a final AI probability.
    Weights can be tuned with validation.
    """
    # Base weights
    w_roberta = 0.5
    w_perplexity = 0.2
    w_semantic = 0.2
    w_heuristics = 0.1

    ai_prob = (roberta_score * w_roberta +
               perplexity_score * w_perplexity +
               semantic_score * w_semantic)

    # Adjust based on heuristics
    if grammar_errors > 5:
        ai_prob *= 0.7  # decrease AI prob if many grammar errors
    if repeated_lines_count > 2:
        ai_prob = min(100, ai_prob + 15)  # increase AI prob if many repeated lines

    return round(ai_prob)


def classify_ai_probability(ai_prob):
    THRESHOLD_AI = 85
    THRESHOLD_HUMAN = 15

    if ai_prob >= THRESHOLD_AI:
        return "AI", "High"
    elif ai_prob <= THRESHOLD_HUMAN:
        return "Human", "High"
    else:
        return "Uncertain", "Moderate"


def translate_text(text, max_len=4900):
    """Translate in chunks to avoid Google Translator limits."""
    if len(text) > max_len:
        chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
        translated_chunks = []
        for chunk in chunks:
            translated_chunks.append(GoogleTranslator(source='auto', target='en').translate(chunk))
        return " ".join(translated_chunks)
    else:
        return GoogleTranslator(source='auto', target='en').translate(text)


@app.get("/")
def home():
    return {"message": "Enhanced AI Text Detector backend running!"}


@app.post("/analyze")
def analyze_text(request: TextRequest):
    text = request.text.strip()
    logger.debug(f"Received text (length {len(text)}): {text[:60]}")

    if len(text) < 20:
        logger.warning("Text too short")
        return {"error": "Text too short. Please provide at least 20 characters."}

    # Detect if input looks like programming code to short-circuit analysis
    if looks_like_code(text):
        logger.info("Programming code detected, skipping AI detection.")
        return {
            "original_text": text,
            "message": "Detected programming code snippet. Skipping AI detection.",
            "ai_probability": "Detected programming code snippet. Skipping AI detection.",
            "label": "Code",
            "confidence": "N/A",
            "strategy": "Regex-based code detection"
        }

    try:
        translated_text = translate_text(text)
        logger.debug(f"Translated text: {translated_text[:60]} ...")
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return {"error": f"Translation failed: {str(e)}"}

    lines = [line.strip() for line in translated_text.split('\n') if line.strip()]
    grammar_errors = grammar_mistakes(translated_text)
    repeated = detect_repeated_lines(lines)
    repeated_lines_count = len(repeated)
    logger.debug(f"Grammar errors: {grammar_errors}, repeated lines count: {repeated_lines_count}")

    roberta_prob = None
    perplexity_prob = None
    semantic_prob = None

    try:
        if model_detector and tokenizer_detector:
            inputs = tokenizer_detector(translated_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model_detector(**inputs)
                scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
                roberta_prob = round(scores[1] * 100, 2)
            logger.debug(f"RoBERTa AI prob: {roberta_prob}")
    except Exception as e:
        logger.error(f"Model inference error: {e}")

    try:
        perplexity = calculate_perplexity(translated_text)
        perplexity_prob = perplexity_to_percentage(perplexity)
        logger.debug(f"Perplexity AI prob: {perplexity_prob}")
    except Exception as e:
        logger.error(f"Perplexity calculation error: {e}")

    try:
        semantic_prob = semantic_similarity_score(translated_text)
        logger.debug(f"Semantic similarity AI prob: {semantic_prob}")
    except Exception as e:
        logger.error(f"Semantic similarity error: {e}")

    # Fallback values if any is None
    roberta_prob = roberta_prob or 50
    perplexity_prob = perplexity_prob or 50
    semantic_prob = semantic_prob or 50

    # Combine all scores and heuristics for final AI probability
    final_ai_prob = combine_scores(roberta_prob, perplexity_prob, semantic_prob, grammar_errors, repeated_lines_count)
    label, confidence = classify_ai_probability(final_ai_prob)

    logger.debug(f"Final classification: {label} with confidence {confidence}")

    return {
        "original_text": text,
        "translated_text": translated_text,
        "ai_probability": f"{final_ai_prob}%",
        "label": label,
        "confidence": confidence,
        "grammar_mistakes": grammar_errors,
        "repeated_lines_count": repeated_lines_count,
        "repeated_lines": repeated,
        "component_scores": {
            "roberta_prob": roberta_prob,
            "perplexity_prob": perplexity_prob,
            "semantic_prob": semantic_prob
        },
        "strategy": "Ensemble RoBERTa + GPT2 + Semantic + Heuristics"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
