import argparse
import os
from rdflib import Graph

def split_nt_file(input_file, chunk_size, output_directory):
  # Splitting an NT file is faster since you don't need to parse it
  with open(input_file, 'r', encoding='utf-8') as infile:
    # Read the file into memory
    print(f"Loading {input_file} into memory...")
    lines = infile.readlines()

  # Get the number of triples in the file
  total_triples = len(lines)
  print(f"Total number of triples: {total_triples}")

  # Calculate the number of chunks needed
  num_chunks = (total_triples + chunk_size - 1) // chunk_size
  print(f"Number of chunks: {num_chunks}")

  for i in range(num_chunks):
    print(f"Processing chunk {i+1} of {num_chunks}...")
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total_triples)

    # Create a new file for the chunk
    chunk_filename = f"chunk_{i+1}.nt"
    chunk_path = f"{output_directory}/{chunk_filename}"

    # Write the chunk to the file
    with open(chunk_path, 'w', encoding='utf-8') as outfile:
      outfile.writelines(lines[start_idx:end_idx])

    print(f"Chunk {i+1} saved to {chunk_path}")


def split_rdf_file(input_file, chunk_size, output_directory, output_format):
  # Load the RDF file
  print(f"Loading {input_file} into memory...")
  g = Graph()
  g.parse(input_file)

  # Get the number of triples in the graph
  total_triples = len(g)
  print(f"Total number of triples: {total_triples}")

  # Calculate the number of chunks needed
  num_chunks = (total_triples + chunk_size - 1) // chunk_size
  print(f"Number of chunks: {num_chunks}")

  for i in range(num_chunks):
    print(f"Processing chunk {i+1} of {num_chunks}...")
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total_triples)

    # Create a new graph for the chunk
    chunk_graph = Graph()

    # Add triples to the chunk graph
    for triple in list(g)[start_idx:end_idx]:
      chunk_graph.add(triple)

    # Save the chunk to an RDF file
    chunk_filename = f"chunk_{i+1}.{output_format}"
    chunk_path = f"{output_directory}/{chunk_filename}"

    # Copy the namespaces from the original graph to the chunk graph
    for prefix, namespace in g.namespaces():
      chunk_graph.bind(prefix, namespace)

    chunk_graph.serialize(chunk_path, format=output_format)
    print(f"Chunk {i+1} saved to {chunk_path}")

def main():
  parser = argparse.ArgumentParser(description="Split an RDF Turtle file into multiple chunks.")
  parser.add_argument("-f", "--format", default="nt", help="Format of the output RDF file (default: nt)")
  parser.add_argument("input_file", help="Path to the input RDF Turtle file")
  parser.add_argument("chunk_size", type=int, help="Number of triples per chunk")

  parser.add_argument("output_directory", help="Directory to save the split RDF chunks")
  args = parser.parse_args()

  # Make sure the output directory exists
  os.makedirs(args.output_directory, exist_ok=True)

  if args.format == "nt":
    split_nt_file(args.input_file, args.chunk_size, args.output_directory)
  else:
    split_rdf_file(args.input_file, args.chunk_size, args.output_directory, args.format)

if __name__ == "__main__":
  main()