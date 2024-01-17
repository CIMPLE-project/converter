import os
import argparse
import requests

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="Path to the RDF file to upload")
args = parser.parse_args()

url = "https://data.cimple.eu/sparql-graph-crud-auth?graph=http://data.cimple.eu/graph/claimreview"
file_path = args.file_path
username = "dba"
password = os.environ["DBA_PASSWORD"]

# Prepare the authentication credentials
auth = requests.auth.HTTPDigestAuth(username, password)

# Prepare the headers
headers = {"Content-Type": "text/turtle"}

# Open and read the file
with open(file_path, "rb") as file:
    # Perform the POST request
    response = requests.post(url, auth=auth, headers=headers, data=file)

# Print the response
print(response.text)
