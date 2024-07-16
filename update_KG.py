import argparse
from rdflib import Graph, Literal, URIRef, XSD
from rdflib.namespace import RDF, FOAF, SDO, Namespace
from tqdm import tqdm, trange
import hashlib
import re
import requests
import json
import io
import html
import torch
import torch.nn as nn
from transformers import BertForPreTraining, AutoTokenizer
import trafilatura

import os

def normalize_text(text):
    text = text.replace('&amp;', '&')
    text = text.replace('\xa0', '')
    text = re.sub(r'http\S+', '', text)
    text = html.unescape(text)
    text = " ".join(text.split())
    return text

def uri_generator(identifier):
    h = hashlib.sha224(str.encode(identifier)).hexdigest()

    return str(h)

def extract_dbpedia_entities(s):
    API_URL = "https://api.dbpedia-spotlight.org/en/annotate"
    if not s in ['', ' ', '   ']:
        try:
            payload = {'text': s}
            a = requests.post(API_URL,
                         headers={'accept': 'application/json'},
                         data=payload).json()
            return a
        except:
            return []
    else:
        return []

def fetch_and_extract_text(url):
    try:
        # Download the URL and extract the main text
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            # Try again with a different user agent
            downloaded = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)'}).text
        main_text = trafilatura.extract(downloaded)
        return main_text
    except Exception as e:
        return str(e)

def compute_factors(text, compute_emotion=True, compute_political_leaning=True, compute_sentiment=True, compute_conspiracy_theories=True):
    emotion, political_bias, sentiment, conspiracy, readability = None, None, None, None, None

    MAX_LEN = 128 # < m some tweets will be truncated
    EMOTIONS = ['None', 'Happiness', 'Anger', 'Sadness', 'Fear']
    POLITICAL_BIAS = ['Left', 'Other', 'Right']
    SENTIMENTS = ['Negative', 'Neutral', 'Positive']
    CONSPIRACIES = ['Suppressed Cures', 'Behaviour and mind Control', 'Antivax', 'Fake virus', 'Intentional Pandemic', 'Harmful Radiation', 'Population Reduction', 'New World Order', 'Satanism']
    CONSPIRACY_LEVELS = ['No ', 'Mentioning ', 'Supporting ']

    a = tokenizer([text], max_length=MAX_LEN, padding='max_length', truncation=True)
    input_ids = torch.tensor(a['input_ids'])
    token_type_ids = torch.tensor(a['token_type_ids'])
    attention_mask = torch.tensor(a['attention_mask'])
    with torch.no_grad():
        if compute_emotion:
            logits_em = model_em(input_ids, token_type_ids, attention_mask)
        if compute_political_leaning:
            logits_pol = model_pol(input_ids, token_type_ids, attention_mask)
        if compute_sentiment:
            logits_sent = model_sent(input_ids, token_type_ids, attention_mask)
        if compute_conspiracy_theories:
            logits_con = model_con(input_ids, token_type_ids, attention_mask)

    emotion = EMOTIONS[logits_em.detach().numpy()[0].argmax()]
    political_bias = POLITICAL_BIAS[logits_pol.detach().numpy()[0].argmax()]
    sentiment = SENTIMENTS[logits_sent.detach().numpy()[0].argmax()]
    conspiracy = [logits_con.detach().numpy()[0][3*i:3*i+3].argmax() for i in range(0, 9)]
    conspiracies = []
    for i in range(0, 9):
        conspiracies.append(CONSPIRACY_LEVELS[conspiracy[i]]+CONSPIRACIES[i])

    return emotion, political_bias, sentiment, conspiracy

class CovidTwitterBertClassifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.bert = BertForPreTraining.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        self.bert.cls.seq_relationship = nn.Linear(1024, n_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, input_mask):
        outputs = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = input_mask)

        logits = outputs[1]
        return logits

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input folder", required=True)
parser.add_argument("-o", "--output", help="Output file", required=True)
parser.add_argument("-g", "--graph", help="Old graph file")
parser.add_argument("-f", "--format", help="Output format", default="nt")
parser.add_argument("-c", "--cache", help="Cache folder", default="cache")
parser.add_argument("-m", "--models", help="Models folder", default="/data/cimple-factors-models")
parser.add_argument("-q", "--quiet", help="Quiet mode", action="store_true")
args = parser.parse_args()

SCHEMA = Namespace("http://schema.org/")
CIMPLE = Namespace("http://data.cimple.eu/ontology#")
SO = Namespace("http://schema.org/")
WIKI_prefix = "http://www.wikidata.org/wiki/"
DB_prefix = "http://dbpedia.org/ontology/"

prefix = "http://data.cimple.eu/"

old_graph = Graph()
new_graph = Graph()

URL_AVAILABLE_CHARS = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;="""
CONSPIRACIES = ['Suppressed Cures', 'Behaviour and mind Control', 'Antivax', 'Fake virus', 'Intentional Pandemic', 'Harmful Radiation', 'Population Reduction', 'New World Order', 'Satanism']

directory = args.input
old_graph_path = args.graph if args.graph else os.path.join(args.cache, 'claim-review.ttl')

print("Loading old graph")
if os.path.exists(os.path.join(args.graph)):
    old_graph.parse(old_graph_path)

print('Loading new Claim Review dataset')
cr_new = json.load(io.open(os.path.join(directory, 'claim_reviews.json')))

print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')

print("Loading emotion model")
model_em = CovidTwitterBertClassifier(5)
model_em.load_state_dict(torch.load(os.path.join(args.models, 'emotion.pth'), map_location=torch.device('cpu')))
model_em.eval()

print("Loading political-leaning model")
model_pol = CovidTwitterBertClassifier(3)
model_pol.load_state_dict(torch.load(os.path.join(args.models, 'political-leaning.pth'), map_location=torch.device('cpu')))
model_pol.eval()

print("Loading sentiment model")
model_sent = CovidTwitterBertClassifier(3)
model_sent.load_state_dict(torch.load(os.path.join(args.models, 'sentiment.pth'), map_location=torch.device('cpu')))
model_sent.eval()

print("Loading conspiracies model")
model_con = CovidTwitterBertClassifier(27)
model_con.load_state_dict(torch.load(os.path.join(args.models, 'conspiracy.pth'), map_location=torch.device('cpu')))
model_con.eval()

need_extract = 0
need_convert = 0

print('Extracting factors, entities and text from new claim reviews')
dict_factors_entities_text = {}
for i in (trange(0, len(cr_new)) if not args.quiet else range(0, len(cr_new))):
    cr_doc = cr_new[i]
    review_text = normalize_text(cr_doc['claim_text'][0])
    if len(review_text)<1:
        errors.append(i)
        continue
    if not cr_doc['reviews'][0]['original_label']:
        errors.append(i)
        continue
    review_label = cr_doc['label']
    review_url = cr_doc['review_url']
    if review_url[-1] !='/':
        review_url += '/'
    review_url_parsed = urlparse(review_url.lower())
    review_url = review_url_parsed.netloc + review_url_parsed.path
    
    review_date = str(cr_doc['reviews'][0]['date_published'])
    identifier = 'claim-review'+review_text+review_label+review_url+review_date
    uri = 'claim-review/'+uri_generator(identifier)

    if (URIRef(prefix+uri), None, None) in old_graph:
        continue

    print(f"Extracting for claim review {i}/{len(cr_new)}: {prefix+uri} ({cr_doc['review_url']})")
    need_extract += 1

    s = review_text

    e, p, s, c, = compute_factors(s)
    factors = {}
    factors['emotion'] = e
    factors['political-leaning'] = p
    factors['sentiment'] = s
    factors['conspiracies'] = c
    factors['readability'] = ''

    a = extract_dbpedia_entities(s)

    factors['entities'] = a
    url = cr_doc['review_url']
    url = url.replace(' ', '')

    url_text = fetch_and_extract_text(url)
    try:
        url_entities = extract_dbpedia_entities(url_text)
    except:
        url_entities = []

    dict_factors_entities_text[uri] = [factors, [url_text, url_entities]]




all_organizations_names = []
all_organizations_websites = []

for cr in cr_new:
    author = cr['fact_checker']['name']
    website = cr['fact_checker']['website']
    if author not in all_organizations_names:
        all_organizations_names.append(author)
        all_organizations_websites.append(website)


errors = []


print('Updating Graph')
for i in (trange(0, len(cr_new)) if not args.quiet else range(0, len(cr_new))):
    cr_doc = cr_new[i]
    review_text = normalize_text(cr_doc['claim_text'][0])
    if len(review_text)<1:
        errors.append(i)
        continue
    if not cr_doc['reviews'][0]['original_label']:
        errors.append(i)
        continue
    review_label = cr_doc['label']
    review_url = cr_doc['review_url']
    if review_url[-1] !='/':
        review_url += '/'
    review_url_parsed = urlparse(review_url.lower())
    review_url = review_url_parsed.netloc + review_url_parsed.path
    
    review_date = str(cr_doc['reviews'][0]['date_published'])
    identifier = 'claim-review'+review_text+review_label+review_url+review_date
    uri = 'claim-review/'+uri_generator(identifier)
    
    # Skip if already converted
    if (URIRef(prefix+uri), None, None) in old_graph:
        continue

    print(f"Converting claim review {i}/{len(cr_new)}: {prefix+uri} ({cr_doc['review_url']})")
    need_convert += 1

    new_graph.add((URIRef(prefix+uri), RDF.type, SCHEMA.ClaimReview))

    author = cr_doc['fact_checker']['name']
    website = cr_doc['fact_checker']['website']
    identifier_author = 'organization'+str(author)
    uri_author = 'organization/'+uri_generator(identifier_author)

    new_graph.add((URIRef(prefix+uri_author), RDF.type, SCHEMA.Organization))
    new_graph.add((URIRef(prefix+uri_author), SCHEMA.name, Literal(author)))
    new_graph.add((URIRef(prefix+uri_author), SCHEMA.url, URIRef(website)))

    new_graph.add((URIRef(prefix+uri), SCHEMA.author, URIRef(prefix+uri_author)))

    date = cr_doc['reviews'][0]['date_published']
    if date:
        new_graph.add((URIRef(prefix+uri), SCHEMA.datePublished, Literal(date, datatype=XSD.date)))

    url = "http://"+review_url
    url = url.replace(' ', '')
    new_graph.add((URIRef(prefix+uri), SCHEMA.url, URIRef(url)))

    cr_text, cr_entities = dict_factors_entities_text[uri][1]
    new_graph.add((URIRef(prefix+uri), SCHEMA.text, Literal(cr_text)))
    if 'Resources' in cr_entities:
        for ent in cr_entities['Resources']:
            dbpedia_url = ent['@URI']
            new_graph.add((URIRef(prefix+uri), SCHEMA.mentions, URIRef(dbpedia_url)))

    language = cr_doc['fact_checker']['language']
    new_graph.add((URIRef(prefix+uri), SCHEMA.inLanguage, Literal(language)))

    uri_normalized_rating = 'rating/'+cr_doc['reviews'][0]['label']
    new_graph.add((URIRef(prefix+uri), CIMPLE.normalizedReviewRating, URIRef(prefix+uri_normalized_rating)))
    new_graph.add((URIRef(prefix+uri_normalized_rating), RDF.type, SCHEMA.Rating))

    uri_original_rating = 'rating/'+uri_generator('rating'+cr_doc['reviews'][0]['original_label'])
    new_graph.add((URIRef(prefix+uri), SCHEMA.reviewRating, URIRef(prefix+uri_original_rating)))
    new_graph.add((URIRef(prefix+uri_original_rating), RDF.type, SCHEMA.Rating))

    claim = cr_doc['claim_text'][0]
    identifier_claim = 'claim'+claim
    uri_claim = 'claim/'+uri_generator(identifier_claim)

    #SCHEMA.Claim has not yet been integrated
    #This term is proposed for full integration into Schema.org, pending implementation feedback and adoption from applications and websites.
    new_graph.add((URIRef(prefix+uri_claim),RDF.type, SCHEMA.Claim))

    new_graph.add((URIRef(prefix+uri), SCHEMA.itemReviewed, URIRef(prefix+uri_claim)))

    text = claim
    text = normalize_text(text)
    new_graph.add((URIRef(prefix+uri_claim),SCHEMA.text, Literal(text)))

    appearances = cr_doc['appearances']
    for a in appearances:
        if a != None:

            b = ''.join([i for i in a if i in URL_AVAILABLE_CHARS])
            new_graph.add((URIRef(prefix+uri_claim), SCHEMA.appearance, URIRef(b)))



    r = dict_factors_entities_text[uri][0]['readability']
    new_graph.add((URIRef(prefix+uri_claim), CIMPLE.readability_score, Literal(r)))


    e = dict_factors_entities_text[uri][0]['emotion']
    if e != 'None':
        new_graph.add((URIRef(prefix+uri_claim), CIMPLE.hasEmotion, URIRef(prefix+'emotion/'+str(e.lower()))))
    s = dict_factors_entities_text[uri][0]['sentiment']
    new_graph.add((URIRef(prefix+uri_claim), CIMPLE.hasSentiment, URIRef(prefix+'sentiment/'+str(s.lower()))))
    b = dict_factors_entities_text[uri][0]['political-leaning']
    new_graph.add((URIRef(prefix+uri_claim), CIMPLE.hasPoliticalLeaning, URIRef(prefix+'political-leaning/'+str(b.lower()))))
    cons_i = dict_factors_entities_text[uri][0]['conspiracies']
    for k in range(0, len(cons_i)):
        if cons_i[k] == 1:
            c = CONSPIRACIES[k]
            new_graph.add((URIRef(prefix+uri_claim), CIMPLE.mentionsConspiracy, URIRef(prefix+'conspiracy/'+str(c.replace(' ', '_').lower()))))
        elif cons_i[k] == 2:
            c = CONSPIRACIES[k]
            new_graph.add((URIRef(prefix+uri_claim), CIMPLE.promotesConspiracy, URIRef(prefix+'conspiracy/'+str(c.replace(' ', '_').lower()))))

    entities = dict_factors_entities_text[uri][0]['entities']
    if 'Resources' in entities:
        for ent in entities['Resources']:
            dbpedia_url = ent['@URI']

            new_graph.add((URIRef(prefix+uri_claim), SCHEMA.mentions, URIRef(dbpedia_url)))

print('Done')
print(len(errors), "documents skipped because of missing claim text and/or original label. Saved to "+str(directory)+"errors.json.")
with open(str(directory)+'errors.json', 'w') as f:
    json.dump(errors, f)

print("Extracted factors, entities and text from " + str(need_extract) + " claim reviews")
print("Converted " + str(need_convert) + " claim reviews")

labels_mapping = json.load(io.open(os.path.join(directory, 'claim_labels_mapping.json')))

print('Adding normalized ratings to graph')
for label in (tqdm(labels_mapping) if not args.quiet else labels_mapping):
    identifier_original_rating = 'rating'+label['original_label']
    uri_original_rating = 'rating/'+uri_generator(identifier_original_rating)

    new_graph.add((URIRef(prefix+uri_original_rating), RDF.type, SO.Rating))
    new_graph.add((URIRef(prefix+uri_original_rating), SO.ratingValue, Literal(label['original_label'])))
    new_graph.add((URIRef(prefix+uri_original_rating), SO.name, Literal(label['original_label'].replace('_', ' '))))

    uri_rating = 'rating/'+label['coinform_label']

    new_graph.add((URIRef(prefix+uri_rating), RDF.type, SO.Rating))
    new_graph.add((URIRef(prefix+uri_rating), SO.ratingValue, Literal(label['coinform_label'])))
    new_graph.add((URIRef(prefix+uri_rating), SO.name, Literal(label['coinform_label'].replace('_', ' '))))

    new_graph.add((URIRef(prefix+uri_original_rating), SO.sameAs, URIRef(prefix+uri_rating)))

    domains = label['domains'].split(',')
    for d in domains:
        corresponding_org_website = None
        for websites in all_organizations_websites:
            if d in websites:
                corresponding_org_website = websites
                break
        if corresponding_org_website is not None:
            corresponding_org_name = all_organizations_names[all_organizations_websites.index(corresponding_org_website)]
            identifier_author = 'organization' + str(corresponding_org_name)
            uri_author = 'organization/' + uri_generator(identifier_author)
            new_graph.add((URIRef(prefix+uri_original_rating), SO.author, URIRef(prefix+uri_author)))

print('Done')

output_file = args.output
print('Nb Nodes:', len(new_graph))
print('Saving new RDF graph to ' + output_file)
new_graph.serialize(destination=output_file, format=args.format, encoding="utf-8")
print('Done')

print('Merging new RDF graph into RDF old graph')
old_graph.parse(output_file)
old_graph.serialize(destination=old_graph_path, format=args.format, encoding="utf-8")
print('Done')
