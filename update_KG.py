import argparse
import hashlib
import html
import io
import json
import logging
import os
import re
from datetime import datetime
from urllib.parse import quote, urlparse

import requests
import torch
import torch.nn as nn
import trafilatura
from rdflib import XSD, Graph, Literal, URIRef
from rdflib.namespace import RDF, Namespace
from rdflib.util import guess_format
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BertForPreTraining

SCHEMA = Namespace("http://schema.org/")
CIMPLE = Namespace("http://data.cimple.eu/ontology#")
WIKI_PREFIX = "http://www.wikidata.org/wiki/"
DB_ONTOLOGY_PREFIX = "http://dbpedia.org/ontology/"
CIMPLE_PREFIX = "http://data.cimple.eu/"

EMOTIONS_LIST = ["None", "Happiness", "Anger", "Sadness", "Fear"]
POLITICAL_BIAS_LIST = ["Left", "Other", "Right"]
SENTIMENTS_LIST = ["Negative", "Neutral", "Positive"]
CONSPIRACIES_LIST = [
    "Suppressed Cures",
    "Behaviour and mind Control",
    "Antivax",
    "Fake virus",
    "Intentional Pandemic",
    "Harmful Radiation",
    "Population Reduction",
    "New World Order",
    "Satanism",
]
CONSPIRACY_LEVELS_LIST = [
    "No ",
    "Mentioning ",
    "Supporting ",
]

MODEL_MAX_LEN = 128  # < m some tweets will be truncated
FACTOR_COMPUTATION_BATCH_SIZE = 32  # Configurable batch size for BERT models
URL_AVAILABLE_CHARS = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;="""

logger = logging.getLogger(__name__)


def normalize_text(text):
    if not isinstance(text, str):  # Ensure text is a string
        return ""
    text = text.replace("&amp;", "&")  # Normalize ampersands
    text = text.replace("\xa0", "")  # Remove non-breaking spaces
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = html.unescape(text)  # Unescape HTML entities
    text = " ".join(text.split())  # Normalize whitespace
    return text


def sanitize_url(url):
    """Sanitize URL by properly encoding invalid URI characters"""
    if not url or not isinstance(url, str):
        return ""

    # Remove leading/trailing whitespace
    url = url.strip()

    # If URL doesn't have a scheme, add http://
    if not url.startswith(("http://", "https://")):
        if "://" not in url:
            url = "http://" + url

    try:
        # Parse the URL
        parsed = urlparse(url)

        # URL encode the path, query, and fragment parts to handle special characters
        safe_path = quote(parsed.path.encode("utf-8"), safe="/:@!$&'()*+,;=")
        safe_query = quote(parsed.query.encode("utf-8"), safe="/:@!$&'()*+,;=?")
        safe_fragment = quote(parsed.fragment.encode("utf-8"), safe="/:@!$&'()*+,;=")

        # Reconstruct the URL with encoded parts
        sanitized_url = f"{parsed.scheme}://{parsed.netloc}{safe_path}"
        if safe_query:
            sanitized_url += f"?{safe_query}"
        if safe_fragment:
            sanitized_url += f"#{safe_fragment}"

        return sanitized_url
    except Exception as e:
        logger.warning(f"Could not sanitize URL '{url}': {e}")
        # Fallback: remove clearly invalid characters
        invalid_chars = ["{", "}", "[", "]", "|", "\\", "^", "`", " "]
        for char in invalid_chars:
            url = url.replace(char, quote(char))
        return url


def uri_generator(identifier):
    h = hashlib.sha224(str.encode(str(identifier))).hexdigest()
    return str(h)


def extract_dbpedia_entities(
    text_content, api_url="https://api.dbpedia-spotlight.org/en/annotate"
):
    if text_content not in ["", " ", "   "]:
        return []
    try:
        payload = {"text": text_content}
        headers = {"accept": "application/json"}
        response = requests.post(api_url, headers=headers, data=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"DBpedia Spotlight API request failed: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from DBpedia Spotlight: {e}")
        return []


def fetch_and_extract_text(url):
    try:
        # Attempt with trafilatura's default fetch
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            # Fallback with a common user-agent
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            downloaded = response.text

        if downloaded:
            main_text = trafilatura.extract(downloaded)
            return main_text if main_text else ""
        return ""
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        return f"Error fetching URL: {e}"
    except Exception as e:
        logger.error(f"Error extracting text from URL {url} with Trafilatura: {e}")
        return f"Error extracting text: {e}"


def compute_factors_batch(texts, tokenizer, models, device):
    model_em, model_pol, model_sent, model_con = models
    results = []

    tokenized_batch = tokenizer(
        texts,
        max_length=MODEL_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenized_batch["input_ids"].to(device)
    token_type_ids = tokenized_batch["token_type_ids"].to(device)
    attention_mask = tokenized_batch["attention_mask"].to(device)

    with torch.no_grad():
        logits_em_batch = model_em(input_ids, token_type_ids, attention_mask)
        logits_pol_batch = model_pol(input_ids, token_type_ids, attention_mask)
        logits_sent_batch = model_sent(input_ids, token_type_ids, attention_mask)
        logits_con_batch = model_con(input_ids, token_type_ids, attention_mask)

    predictions_em = logits_em_batch.detach().cpu().numpy().argmax(axis=1)
    predictions_pol = logits_pol_batch.detach().cpu().numpy().argmax(axis=1)
    predictions_sent = logits_sent_batch.detach().cpu().numpy().argmax(axis=1)

    # For conspiracies, shape is (batch_size, num_conspiracies * num_levels)
    # Reshape and then argmax for each conspiracy type
    num_conspiracies = len(CONSPIRACIES_LIST)  # Should be 9
    num_levels = len(CONSPIRACY_LEVELS_LIST)  # Should be 3

    predictions_con_reshaped = (
        logits_con_batch.detach()
        .cpu()
        .numpy()
        .reshape(-1, num_conspiracies, num_levels)
    )
    predictions_con = predictions_con_reshaped.argmax(axis=2)

    for i in range(len(texts)):
        emotion = EMOTIONS_LIST[predictions_em[i]]
        political_bias = POLITICAL_BIAS_LIST[predictions_pol[i]]
        sentiment = SENTIMENTS_LIST[predictions_sent[i]]
        conspiracy_indices = list(predictions_con[i])

        results.append(
            {
                "emotion": emotion,
                "political-leaning": political_bias,
                "sentiment": sentiment,
                "conspiracies": conspiracy_indices,  # list of 9 indices (0, 1, or 2)
                "readability": "",
            }
        )
    return results


class CovidTwitterBertClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(
            "digitalepidemiologylab/covid-twitter-bert-v2"
        )
        original_in_features = self.bert.cls.seq_relationship.in_features
        self.bert.cls.seq_relationship = nn.Linear(original_in_features, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, input_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )
        logits = outputs[1]
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    parser.add_argument(
        "-o", "--output", help="Output RDF file for the new triples", required=True
    )
    parser.add_argument(
        "-g",
        "--graph",
        help="Path to the existing RDF graph file ",
    )
    parser.add_argument("-f", "--format", help="Output RDF format", default="nt")
    parser.add_argument(
        "-m", "--models", help="Models folder", default="/data/cimple-factors-models"
    )
    parser.add_argument(
        "--device",
        help="Device for PyTorch models (e.g., 'cpu', 'cuda', 'cuda:0')",
        default="auto",
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for factor computation",
        type=int,
        default=FACTOR_COMPUTATION_BATCH_SIZE,
    )
    parser.add_argument("--no-progress", help="Hide progress bars", action="store_true")
    args = parser.parse_args()

    # Create logs folder based on output directory and current datetime
    output_dir = os.path.dirname(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_folder = os.path.join(output_dir, "logs", timestamp)
    os.makedirs(logs_folder, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logs_folder, "update_kg.log")),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"Created logs folder: {logs_folder}")

    # Determine PyTorch device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using PyTorch device: {device}")

    # Initialize graphs
    old_graph = Graph()
    new_graph = Graph()

    # Load existing URIs from old graph for deduplication
    existing_uris_in_old_graph = set()  # URIs cache
    old_graph_path = args.graph
    if old_graph_path and os.path.exists(old_graph_path):
        logger.info(f"Loading URIs from existing graph: {old_graph_path}")
        try:
            old_graph.parse(old_graph_path, format=guess_format(old_graph_path))
            for s, _, _ in old_graph:
                if isinstance(s, URIRef):
                    existing_uris_in_old_graph.add(str(s))
            logger.info(
                f"Loaded {len(existing_uris_in_old_graph)} URIs from old graph."
            )
        except Exception as e:
            logger.warning(
                f"Could not parse old graph at {old_graph_path}: {e}. Proceeding without it for deduplication."
            )
            old_graph = Graph()  # Reset if parsing failed
    else:
        logger.info(
            "No existing graph provided or found. All new items will be processed."
        )

    # Load new claim review dataset
    claim_reviews_path = os.path.join(args.input, "claim_reviews.json")
    logger.info(f"Loading new Claim Review dataset from: {claim_reviews_path}")
    try:
        with io.open(claim_reviews_path, "r", encoding="utf-8") as f:
            cr_new = json.load(f)
    except FileNotFoundError:
        logger.error(f"claim_reviews.json not found in {args.input}. Exiting.")
        return
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {claim_reviews_path}. Exiting.")
        return

    # Load tokenizer and models
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "digitalepidemiologylab/covid-twitter-bert"
    )

    models_path = args.models
    logger.info("Loading models...")
    try:
        model_em = CovidTwitterBertClassifier(len(EMOTIONS_LIST)).to(device)
        model_em.load_state_dict(
            torch.load(os.path.join(models_path, "emotion.pth"), map_location=device)
        )
        model_em.eval()

        model_pol = CovidTwitterBertClassifier(len(POLITICAL_BIAS_LIST)).to(device)
        model_pol.load_state_dict(
            torch.load(
                os.path.join(models_path, "political-leaning.pth"), map_location=device
            )
        )
        model_pol.eval()

        model_sent = CovidTwitterBertClassifier(len(SENTIMENTS_LIST)).to(device)
        model_sent.load_state_dict(
            torch.load(os.path.join(models_path, "sentiment.pth"), map_location=device)
        )
        model_sent.eval()

        model_con = CovidTwitterBertClassifier(
            len(CONSPIRACIES_LIST) * len(CONSPIRACY_LEVELS_LIST)
        ).to(
            device
        )  # 9 conspiracies * 3 levels = 27 classes
        model_con.load_state_dict(
            torch.load(os.path.join(models_path, "conspiracy.pth"), map_location=device)
        )
        model_con.eval()
    except FileNotFoundError as e:
        logger.error(
            f"Error loading models: {e}. Ensure models are in '{models_path}'. Exiting."
        )
        return

    bert_models = (model_em, model_pol, model_sent, model_con)

    # Main processing loop
    errors_skipped_docs = []
    processed_count = 0
    items_for_processing_buffer = (
        []
    )  # Buffer for batching factor computation & other per-item tasks

    # For collecting unique organization data
    organizations_data = {}  # name -> {website: "url", uri: "cimple_uri"}

    logger.info(
        "Processing new claim reviews, extracting factors, and building graph..."
    )
    for i in (
        trange(len(cr_new), desc="Claim reviews")
        if not args.no_progress
        else range(len(cr_new))
    ):
        cr_doc = cr_new[i]

        # Basic validation
        if not cr_doc.get("claim_text") or not cr_doc["claim_text"][0]:
            errors_skipped_docs.append({"index": i, "reason": "Missing claim_text"})
            continue
        review_text_original = cr_doc["claim_text"][0]
        review_text_normalized = normalize_text(review_text_original)
        if len(review_text_normalized) < 1:
            errors_skipped_docs.append(
                {"index": i, "reason": "Empty normalized claim_text"}
            )
            continue

        if not cr_doc.get("reviews") or not cr_doc["reviews"][0].get("original_label"):
            errors_skipped_docs.append({"index": i, "reason": "Missing original_label"})
            continue

        # URI Generation
        review_label = cr_doc["label"]
        review_url_raw = cr_doc["review_url"]

        # Normalize review_url for URI generation
        parsed_review_url_for_id = urlparse(review_url_raw.lower())
        review_url_for_id_path = parsed_review_url_for_id.path
        if review_url_for_id_path and not review_url_for_id_path.endswith("/"):
            review_url_for_id_path += "/"
        review_url_normalized_for_id = (
            parsed_review_url_for_id.netloc + review_url_for_id_path
        )

        review_date = str(cr_doc["reviews"][0].get("date_published", ""))
        identifier = (
            "claim-review"
            + review_text_normalized
            + review_label
            + review_url_normalized_for_id
            + review_date
        )
        item_uri_suffix = "claim-review/" + uri_generator(identifier)
        item_full_uri = CIMPLE_PREFIX + item_uri_suffix

        if item_full_uri in existing_uris_in_old_graph:
            continue  # Skip if already processed and in old graph

        # Collect organization data
        org_name = cr_doc["fact_checker"]["name"]
        org_website = cr_doc["fact_checker"]["website"]
        if org_name and org_name not in organizations_data:
            org_identifier = "organization" + str(org_name)
            org_uri_suffix = "organization/" + uri_generator(org_identifier)
            organizations_data[org_name] = {
                "website": org_website,
                "uri_suffix": org_uri_suffix,
            }

        # Add item to buffer for batch processing
        items_for_processing_buffer.append(
            {
                "doc": cr_doc,
                "uri_suffix": item_uri_suffix,
                "claim_text_normalized": review_text_normalized,
                "claim_text_original": review_text_original,
                "review_url_raw": review_url_raw,
                "index": i,
            }
        )

        # Process buffer when it's full or at the end of the input list
        if len(items_for_processing_buffer) >= args.batch_size or i == len(cr_new) - 1:
            if not items_for_processing_buffer:
                continue

            texts_for_factors = [
                item["claim_text_normalized"] for item in items_for_processing_buffer
            ]
            computed_factors_batch = compute_factors_batch(
                texts_for_factors, tokenizer, bert_models, device
            )

            for idx, item_data in enumerate(items_for_processing_buffer):
                cr_doc_item = item_data["doc"]
                current_uri_suffix = item_data["uri_suffix"]
                current_full_uri = CIMPLE_PREFIX + current_uri_suffix
                claim_text_norm = item_data["claim_text_normalized"]
                current_index = item_data["index"]

                factors = computed_factors_batch[idx]

                # 1. Fetch and extract text from review_url
                page_url_to_fetch = item_data["review_url_raw"].replace(
                    " ", ""
                )  # Sanitize spaces
                fetched_url_text = fetch_and_extract_text(page_url_to_fetch)

                # 2. Extract DBpedia entities from fetched_url_text
                url_text_entities_raw = (
                    extract_dbpedia_entities(fetched_url_text)
                    if fetched_url_text
                    else []
                )

                # 3. Extract DBpedia entities from claim_text_normalized
                claim_text_entities_raw = extract_dbpedia_entities(claim_text_norm)

                # Add to new_graph
                new_graph.add((URIRef(current_full_uri), RDF.type, SCHEMA.ClaimReview))

                # Author (organization)
                fact_checker_name = cr_doc_item["fact_checker"]["name"]
                if fact_checker_name in organizations_data:
                    org_uri_item_suffix = organizations_data[fact_checker_name][
                        "uri_suffix"
                    ]
                    org_full_uri = CIMPLE_PREFIX + org_uri_item_suffix
                    new_graph.add((URIRef(org_full_uri), RDF.type, SCHEMA.Organization))
                    new_graph.add(
                        (URIRef(org_full_uri), SCHEMA.name, Literal(fact_checker_name))
                    )

                    # Sanitize organization website URL
                    org_website_sanitized = sanitize_url(
                        organizations_data[fact_checker_name]["website"]
                    )
                    if org_website_sanitized:
                        new_graph.add(
                            (
                                URIRef(org_full_uri),
                                SCHEMA.url,
                                URIRef(org_website_sanitized),
                            )
                        )
                    new_graph.add(
                        (URIRef(current_full_uri), SCHEMA.author, URIRef(org_full_uri))
                    )

                # Date published
                date_pub_str = str(cr_doc_item["reviews"][0].get("date_published"))
                if date_pub_str and date_pub_str != "None":
                    try:
                        new_graph.add(
                            (
                                URIRef(current_full_uri),
                                SCHEMA.datePublished,
                                Literal(date_pub_str, datatype=XSD.date),
                            )
                        )
                    except ValueError:
                        logger.warning(
                            f"Could not parse date '{date_pub_str}' for {current_full_uri}. Adding as plain literal."
                        )
                        new_graph.add(
                            (
                                URIRef(current_full_uri),
                                SCHEMA.datePublished,
                                Literal(date_pub_str),
                            )
                        )

                # Review URL (schema:url for ClaimReview)
                actual_review_url_for_graph = sanitize_url(cr_doc_item["review_url"])
                if actual_review_url_for_graph:
                    new_graph.add(
                        (
                            URIRef(current_full_uri),
                            SCHEMA.url,
                            URIRef(actual_review_url_for_graph),
                        )
                    )

                # Text of the reviewed page (schema:text for ClaimReview)
                if fetched_url_text and not fetched_url_text.startswith("Error"):
                    new_graph.add(
                        (
                            URIRef(current_full_uri),
                            SCHEMA.text,
                            Literal(fetched_url_text),
                        )
                    )

                # Mentions in the reviewed page (schema:mentions for ClaimReview)
                if url_text_entities_raw and "Resources" in url_text_entities_raw:
                    for ent in url_text_entities_raw["Resources"]:
                        if "@URI" in ent:
                            new_graph.add(
                                (
                                    URIRef(current_full_uri),
                                    SCHEMA.mentions,
                                    URIRef(ent["@URI"]),
                                )
                            )

                # Language
                language = cr_doc_item["fact_checker"].get("language")
                if language:
                    new_graph.add(
                        (URIRef(current_full_uri), SCHEMA.inLanguage, Literal(language))
                    )

                # Ratings
                normalized_rating_val = cr_doc_item["reviews"][0]["label"]
                uri_normalized_rating_suffix = "rating/" + normalized_rating_val
                new_graph.add(
                    (
                        URIRef(current_full_uri),
                        CIMPLE.normalizedReviewRating,
                        URIRef(CIMPLE_PREFIX + uri_normalized_rating_suffix),
                    )
                )
                new_graph.add(
                    (
                        URIRef(CIMPLE_PREFIX + uri_normalized_rating_suffix),
                        RDF.type,
                        SCHEMA.Rating,
                    )
                )

                original_rating_val = cr_doc_item["reviews"][0]["original_label"]
                uri_original_rating_suffix = "rating/" + uri_generator(
                    "rating" + original_rating_val
                )
                new_graph.add(
                    (
                        URIRef(current_full_uri),
                        SCHEMA.reviewRating,
                        URIRef(CIMPLE_PREFIX + uri_original_rating_suffix),
                    )
                )
                new_graph.add(
                    (
                        URIRef(CIMPLE_PREFIX + uri_original_rating_suffix),
                        RDF.type,
                        SCHEMA.Rating,
                    )
                )

                # Claim (itemReviewed)
                claim_text_for_node = item_data["claim_text_original"]
                identifier_claim = "claim" + claim_text_for_node
                uri_claim_suffix = "claim/" + uri_generator(identifier_claim)
                claim_full_uri = CIMPLE_PREFIX + uri_claim_suffix

                new_graph.add((URIRef(claim_full_uri), RDF.type, SCHEMA.Claim))
                new_graph.add(
                    (
                        URIRef(current_full_uri),
                        SCHEMA.itemReviewed,
                        URIRef(claim_full_uri),
                    )
                )
                new_graph.add(
                    (
                        URIRef(claim_full_uri),
                        SCHEMA.text,
                        Literal(normalize_text(claim_text_for_node)),
                    )
                )

                # Appearances of the claim
                appearances = cr_doc_item.get("appearances", [])
                for appr_url in appearances:
                    if appr_url and isinstance(appr_url, str):
                        sanitized_appr_url = sanitize_url(appr_url)
                        if sanitized_appr_url:
                            new_graph.add(
                                (
                                    URIRef(claim_full_uri),
                                    SCHEMA.appearance,
                                    URIRef(sanitized_appr_url),
                                )
                            )

                # Factors associated with the Claim
                if factors["readability"]:
                    new_graph.add(
                        (
                            URIRef(claim_full_uri),
                            CIMPLE.readability_score,
                            Literal(factors["readability"]),
                        )
                    )

                if factors["emotion"] != "None":
                    new_graph.add(
                        (
                            URIRef(claim_full_uri),
                            CIMPLE.hasEmotion,
                            URIRef(
                                CIMPLE_PREFIX + "emotion/" + factors["emotion"].lower()
                            ),
                        )
                    )

                new_graph.add(
                    (
                        URIRef(claim_full_uri),
                        CIMPLE.hasSentiment,
                        URIRef(
                            CIMPLE_PREFIX + "sentiment/" + factors["sentiment"].lower()
                        ),
                    )
                )
                new_graph.add(
                    (
                        URIRef(claim_full_uri),
                        CIMPLE.hasPoliticalLeaning,
                        URIRef(
                            CIMPLE_PREFIX
                            + "political-leaning/"
                            + factors["political-leaning"].lower()
                        ),
                    )
                )

                conspiracy_indices = factors["conspiracies"]
                for k_idx, consp_level_idx in enumerate(conspiracy_indices):
                    consp_name = CONSPIRACIES_LIST[k_idx]
                    consp_uri_name = consp_name.replace(" ", "_").lower()
                    if consp_level_idx == 1:  # Mentioning
                        new_graph.add(
                            (
                                URIRef(claim_full_uri),
                                CIMPLE.mentionsConspiracy,
                                URIRef(CIMPLE_PREFIX + "conspiracy/" + consp_uri_name),
                            )
                        )
                    elif consp_level_idx == 2:  # Supporting
                        new_graph.add(
                            (
                                URIRef(claim_full_uri),
                                CIMPLE.promotesConspiracy,
                                URIRef(CIMPLE_PREFIX + "conspiracy/" + consp_uri_name),
                            )
                        )

                # DBpedia entities from the claim_text itself
                if claim_text_entities_raw and "Resources" in claim_text_entities_raw:
                    for ent in claim_text_entities_raw["Resources"]:
                        if "@URI" in ent:
                            new_graph.add(
                                (
                                    URIRef(claim_full_uri),
                                    SCHEMA.mentions,
                                    URIRef(ent["@URI"]),
                                )
                            )

                processed_count += 1
                logger.info(
                    f"Processed {current_index + 1}/{len(cr_new)} - URI: {current_full_uri} - URL: {item_data['review_url_raw']}"
                )

            items_for_processing_buffer.clear()  # Clear buffer after processing

    logger.info(
        f"Processed and added {processed_count} new claim reviews to the new graph."
    )
    if errors_skipped_docs:
        error_file_path = os.path.join(logs_folder, "skipped_documents_errors.json")
        logger.warning(
            f"{len(errors_skipped_docs)} documents skipped. See '{error_file_path}' for details."
        )
        with open(error_file_path, "w") as f:
            json.dump(errors_skipped_docs, f, indent=2)

    # Process claim labels mapping ---
    labels_mapping_path = os.path.join(args.input, "claim_labels_mapping.json")
    if os.path.exists(labels_mapping_path):
        logger.info("Loading and adding normalized ratings from labels mapping...")
        with io.open(labels_mapping_path, "r", encoding="utf-8") as f:
            labels_mapping = json.load(f)

        # Pre-build a map for domain to organization URI for faster lookup
        domain_to_org_uri_map = {}
        for org_name_map, org_details_map in organizations_data.items():
            parsed_org_website = urlparse(org_details_map["website"])
            org_domain = parsed_org_website.netloc.lower().replace("www.", "")
            if org_domain and org_domain not in domain_to_org_uri_map:
                domain_to_org_uri_map[org_domain] = (
                    CIMPLE_PREFIX + org_details_map["uri_suffix"]
                )

        for label_entry in (
            tqdm(labels_mapping, desc="Label mapping")
            if not args.no_progress
            else labels_mapping
        ):
            original_label_val = label_entry.get("original_label")
            coinform_label_val = label_entry.get("coinform_label")

            uri_original_rating_entry_suffix = None
            if original_label_val:
                uri_original_rating_entry_suffix = "rating/" + uri_generator(
                    "rating" + original_label_val
                )
                orig_rating_full_uri = CIMPLE_PREFIX + uri_original_rating_entry_suffix
                new_graph.add((URIRef(orig_rating_full_uri), RDF.type, SCHEMA.Rating))
                new_graph.add(
                    (
                        URIRef(orig_rating_full_uri),
                        SCHEMA.ratingValue,
                        Literal(original_label_val),
                    )
                )
                new_graph.add(
                    (
                        URIRef(orig_rating_full_uri),
                        SCHEMA.name,
                        Literal(original_label_val.replace("_", " ")),
                    )
                )

            if coinform_label_val:
                uri_coinform_rating_suffix = "rating/" + coinform_label_val
                coinform_rating_full_uri = CIMPLE_PREFIX + uri_coinform_rating_suffix
                new_graph.add(
                    (URIRef(coinform_rating_full_uri), RDF.type, SCHEMA.Rating)
                )
                new_graph.add(
                    (
                        URIRef(coinform_rating_full_uri),
                        SCHEMA.ratingValue,
                        Literal(coinform_label_val),
                    )
                )
                new_graph.add(
                    (
                        URIRef(coinform_rating_full_uri),
                        SCHEMA.name,
                        Literal(coinform_label_val.replace("_", " ")),
                    )
                )

                if (
                    uri_original_rating_entry_suffix
                ):  # Link original rating to Coinform rating if both exist
                    new_graph.add(
                        (
                            URIRef(CIMPLE_PREFIX + uri_original_rating_entry_suffix),
                            SCHEMA.sameAs,
                            URIRef(coinform_rating_full_uri),
                        )
                    )

            # Associate rating with authoring organizations based on domain
            domains_str = label_entry.get("domains", "")
            if (
                domains_str and uri_original_rating_entry_suffix
            ):  # Link author to original rating URI
                domain_list = [
                    d.strip().lower().replace("www.", "")
                    for d in domains_str.split(",")
                    if d.strip()
                ]
                for d_clean in domain_list:
                    if d_clean in domain_to_org_uri_map:
                        org_author_uri = domain_to_org_uri_map[d_clean]
                        new_graph.add(
                            (
                                URIRef(
                                    CIMPLE_PREFIX + uri_original_rating_entry_suffix
                                ),
                                SCHEMA.author,
                                URIRef(org_author_uri),
                            )
                        )
    else:
        logger.warning(
            f"Labels mapping file not found at {labels_mapping_path}. Skipping this step."
        )

    logger.info("Done processing labels mapping.")

    # Serialize new graph and merge
    output_file_path = args.output
    logger.info(f"Nb Triples in new graph: {len(new_graph)}")
    logger.info(f"Saving new RDF graph to: {output_file_path}")
    new_graph.serialize(
        destination=output_file_path, format=args.format, encoding="utf-8"
    )
    logger.info("Done saving new graph.")

    if old_graph_path and os.path.exists(old_graph_path):
        logger.info("Merging new RDF graph into old RDF graph in memory...")
        for triple in new_graph:
            old_graph.add(triple)

        logger.info(f"Nb Triples in merged graph: {len(old_graph)}")
        logger.info(f"Saving merged RDF graph to: {old_graph_path}")
        old_graph.serialize(
            destination=old_graph_path, format=args.format, encoding="utf-8"
        )
        logger.info("Done merging and saving.")
    else:
        logger.info(
            "No existing graph path provided or it was invalid; the new graph is saved, but no merge performed."
        )

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
