import argparse
from rdflib import Graph

def split_rdf_file(input_file, chunk_size, output_directory, format):
  # Load the RDF Turtle file
  print(f"Loading {input_file} into memory...")
  g = Graph()
  g.parse(input_file, format="turtle")

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
    chunk_filename = f"chunk_{i+1}.{format}"
    chunk_path = f"{output_directory}/{chunk_filename}"

    # Copy the namespaces from the original graph to the chunk graph
    for prefix, namespace in g.namespaces():
      chunk_graph.bind(prefix, namespace)

    chunk_graph.serialize(chunk_path, format=format)
    print(f"Chunk {i+1} saved to {chunk_path}")

def main():
  parser = argparse.ArgumentParser(description="Split an RDF Turtle file into multiple chunks.")
  parser.add_argument("-f", "--format", default="nt", help="Format of the output RDF file (default: nt)")
  parser.add_argument("input_file", help="Path to the input RDF Turtle file")
  parser.add_argument("chunk_size", type=int, help="Number of triples per chunk")

  parser.add_argument("output_directory", help="Directory to save the split RDF chunks")
  args = parser.parse_args()

  split_rdf_file(args.input_file, args.chunk_size, args.output_directory, args.format)

if __name__ == "__main__":
  main()