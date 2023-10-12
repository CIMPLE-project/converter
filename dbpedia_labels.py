import os
import argparse
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS
from tqdm import tqdm

# Define the RDF namespaces
SCHEMA = Namespace("http://schema.org/")

def create_dbpedia_label_map(labels_file):
    print("Parsing", labels_file)
    # Load and parse the labels file
    labels_graph = Graph()
    labels_graph.parse(labels_file, format="turtle")
    print("Done parsing, extracting labels...")

    # Create a dictionary to store the DBpedia URI -> Label mapping
    dbpedia_label_map = {}

    # Extract DBpedia URIs and their labels
    for uri, _, label_literal in labels_graph.triples((None, RDFS.label, None)):
        uri_str = str(uri)
        dbpedia_label_map[uri_str] = label_literal

    return dbpedia_label_map

def find_mentions_in_rdf_files(input_path, output_file, labels_file):
    if not os.path.exists(input_path) or not os.path.exists(labels_file):
        print("Invalid path. Please provide valid folder and labels file paths.")
        return

    # Create a DBpedia URI -> Label map
    dbpedia_label_map = create_dbpedia_label_map(labels_file)

    # Initialize an RDF graph to store the data
    rdf_graph = Graph()
    output_graph = Graph()

    # Recursively scan the input directory for RDF files
    rdf_files = [os.path.join(foldername, filename) for foldername, _, filenames in os.walk(input_path) for filename in filenames if filename.endswith(".ttl")]

    # Initialize tqdm with the total number of RDF files
    with tqdm(total=len(rdf_files), desc="Processing RDF files") as pbar:
        for rdf_file_path in rdf_files:
            # Load the RDF file into the graph
            rdf_graph.parse(rdf_file_path)
            pbar.update(1)  # Increment progress

    # Find and print all values of schema:mentions
    for _, _, obj in rdf_graph.triples((None, SCHEMA.mentions, None)):
        mention = str(obj)
        if mention in dbpedia_label_map:
            output_graph.add((obj, RDFS.label, dbpedia_label_map[mention]))

    # Serialize the labels graph to the specified Turtle file
    output_graph.serialize(destination=output_file, format="turtle")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DBpedia URIs and their English labels from RDF files")
    parser.add_argument("-i", "--input", help="Path to the folder containing RDF files")
    parser.add_argument("-l", "--labels", help="Path to the labels TTL file")
    parser.add_argument("-o", "--output", help="Path to the output Turtle file")
    args = parser.parse_args()

    find_mentions_in_rdf_files(args.input_path, args.output_file, args.labels_file)