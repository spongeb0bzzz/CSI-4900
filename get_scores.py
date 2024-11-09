import pandas as pd
import requests
import re
import json
from urllib.parse import urlparse
import joblib
import pandas as pd
import faiss

######################################################################
# CODE FOR CHECK IF LINK IS IN DATABASE
######################################################################

page_ranking_df = pd.read_csv("https://media.githubusercontent.com/media/spongeb0bzzz/CSI-4900/refs/heads/main/data/top10milliondomains.csv")
page_ranking_df.columns = page_ranking_df.columns.str.strip()  # Clean column names
page_ranking_df.set_index('Domain', inplace=True)  # Set index to 'Domain'

def clean_link(link):
    # Validate if link is in a correct URL format
    try:
        if not link.startswith(('http://', 'https://')):
            link = 'http://' + link  # Prepend a scheme if missing
        parsed_url = urlparse(link)
        netloc = parsed_url.netloc
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return netloc
    except Exception as e:
        print(f"Skipping invalid link: {link} - Error: {e}")
        return None

# Function to download and parse the phishing links
def load_and_process_sources(clean=True):
    # Load the first dataset (combined phishing and benign URLs)
    combined_urls_url = "https://raw.githubusercontent.com/spongeb0bzzz/CSI-4900/refs/heads/main/data/combined_urls.csv"
    combined_urls = pd.read_csv(combined_urls_url)

    if(clean):
        # Clean the "link" column (remove scheme and subdomain) and remove null values
        combined_urls['cleaned_link'] = combined_urls['link'].apply(clean_link)
        combined_urls.dropna(subset=['cleaned_link'], inplace=True)  # Remove rows with None in 'cleaned_link'

        # Create a set of cleaned links with their status (0 for benign, 1 for phishing)
        combined_urls_set = set(zip(combined_urls['cleaned_link'], combined_urls['status']))

    else:
        combined_urls['cleaned_link'] = combined_urls['link']
        combined_urls.dropna(subset=['cleaned_link'], inplace=True)  # Remove rows with None in 'cleaned_link'
        combined_urls_set = set(zip(combined_urls['cleaned_link'], combined_urls['status']))

    # Load the second set of sources (multiple files from GitHub)
    sources = [
        "https://raw.githubusercontent.com/phishfort/phishfort-lists/refs/heads/master/blacklists/domains.json",
        "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/refs/heads/master/phishing-links-ACTIVE.txt",
        "https://raw.githubusercontent.com/romainmarcoux/malicious-domains/refs/heads/main/full-domains-aa.txt",
        "https://raw.githubusercontent.com/romainmarcoux/malicious-domains/refs/heads/main/full-domains-aa.txt.txt",
        "https://raw.githubusercontent.com/romainmarcoux/malicious-domains/refs/heads/main/full-domains-ab.txt",
        "https://raw.githubusercontent.com/romainmarcoux/malicious-domains/refs/heads/main/full-domains-ab.txt.txt"
    ]

    # Process each source
    for source_url in sources:
        if source_url.endswith('.json'):
            # If the source is a JSON file, load it as a list
            response = requests.get(source_url)
            domains = json.loads(response.text)
            for domain in domains:
                cleaned_domain = clean_link(domain)
                combined_urls_set.add((cleaned_domain, 1))  # All entries from these sources are phishing
        else:
            # If the source is a text file, load it line by line
            response = requests.get(source_url)
            for line in response.text.splitlines():
                cleaned_domain = clean_link(line)
                combined_urls_set.add((cleaned_domain, 1))  # All entries from these sources are phishing

    return pd.DataFrame(list(combined_urls_set), columns=['link', 'status'])

def clean_dataset(dataset):
  regex = r"(?:https?:\/\/[a-zA-Z0-9.-]+|ftp:\/\/[a-zA-Z0-9.-]+|\/\/[a-zA-Z0-9.-]+|www\.[a-zA-Z0-9.-]+|[a-zA-Z0-9-]+\.[a-zA-Z]{1,}|(?:\d{1,3}\.){3}\d{1,3})(?:[\/a-zA-Z0-9.-]*)[^\s<>,\'\"\)]*"

  # Apply the regex to the 'link' column to capture valid URLs
  valid_links = dataset['link'].str.contains(regex, regex=True)

  # Count the number of valid links
  num_valid_links = valid_links.sum()
  total_links = dataset.shape[0]

  # Calculate the percentage of captured links
  percent_captured = (num_valid_links / total_links) * 100 if total_links > 0 else 0

  # Print the results
#   print(f"Number of valid links: {num_valid_links}")
#   print(f"Total links: {total_links}")
#   print(f"Percentage of captured links: {percent_captured:.2f}%")

  valid_links = dataset['link'].str.contains(regex, regex=True)

  # Filter the dataset to get links that weren't captured by the regex
  uncaptured_links = dataset[~valid_links]

  # Set pandas to display all rows
  pd.set_option('display.max_rows', None)

  # Print the entire DataFrame of uncaptured links
  print("Links not captured by the regex:")
  print(uncaptured_links)

  # Keep only the captured links
  dataset = dataset[valid_links]

  # Mapping string values to integers in the 'status' column
  dataset['status'] = dataset['status'].replace({
      '1': 1,
      '0': 0,
      'malware': 1
  })


  # Optionally, reset the index if desired
  dataset.reset_index(drop=True, inplace=True)

  return dataset

combined_df = load_and_process_sources()
combined_df = clean_dataset(combined_df)
combined_df.set_index('link', inplace=True)  # Set index to 'link'

def get_result_from_database(link):
    '''
    Returns 1 (100%) if the link is phishing
    Returns 0 (0%) if the link is benign
    Returns None if the link is not found
    '''
    cleaned_link = clean_link(link)

    # Check if the link exists in the page_ranking_df (benign domains)
    if cleaned_link in page_ranking_df.index:
        return 0  # Return 0 for benign if the domain is found in the page ranking dataset

    # Check if the link exists in the combined_df (phishing dataset)
    if cleaned_link in combined_df.index:
        return combined_df.loc[cleaned_link, 'status']  # Return the status (0 for benign, 1 for phishing)

    return None  # If the link is not found in either dataset, return None

######################################################################
# CODE FOR CHECKING IP ADDRESS
######################################################################

def extract_ips_from_url(url):
    response = requests.get(url)
    # Split the content into lines and filter out any non-IP lines
    ip_addresses = {line.strip() for line in response.text.splitlines() if re.match(r'^\d+\.\d+\.\d+\.\d+$', line.strip())}
    return ip_addresses

def combine_ip_lists():
  urls = [
    "https://raw.githubusercontent.com/bitwire-it/ipblocklist/refs/heads/main/ip-list.txt",
    "https://raw.githubusercontent.com/duggytuxy/malicious_ip_addresses/refs/heads/main/blacklist_ips_for_fortinet_firewall_aa.txt",
    "https://raw.githubusercontent.com/duggytuxy/malicious_ip_addresses/refs/heads/main/blacklist_ips_for_fortinet_firewall_ab.txt",
    "https://raw.githubusercontent.com/duggytuxy/malicious_ip_addresses/refs/heads/main/botnets_zombies_scanner_spam_ips.txt",
    "https://raw.githubusercontent.com/romainmarcoux/malicious-outgoing-ip/refs/heads/main/full-outgoing-ip-40k.txt",
    "https://raw.githubusercontent.com/romainmarcoux/malicious-outgoing-ip/refs/heads/main/full-outgoing-ip-aa.txt",
    "https://raw.githubusercontent.com/romainmarcoux/malicious-outgoing-ip/refs/heads/main/full-outgoing-ip-ab.txt"
  ]

  # Combine all the malicious IP addresses into a single set
  malicious_ips = set()

  for url in urls:
      malicious_ips.update(extract_ips_from_url(url))

  return malicious_ips

malicious_ips = combine_ip_lists()

def check_sender_ip(sender_ip):
    # Check if the sender's IP is in the malicious IPs set
    if sender_ip in malicious_ips:
        return 1  # Malicious IP
    else:
        return 0  # Benign IP

######################################################################
# CODE FOR CASE-BASED REASONNING
######################################################################

vectorizer = joblib.load("")
faiss_index = faiss.read_index("")
links = joblib.load("")
statuses = joblib.load("")

def get_cbr_score(link, top_k=1):
    query_vector = vectorizer.transform([link]).toarray().astype('float32')
    distances, indices = faiss_index.search(query_vector, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        similarity_score = 100 - distances[0][i]  # Distance to similarity percentage
        matched_link = links[idx]
        status = 1 if statuses[idx] == 1 else 0  # 1 for phishing, 0 for benign

        results.append({
            "similarity_score": similarity_score,
            "matched_link": matched_link,
            "status": status
        })

    return results
# Example to get similarity score: results[0]['similarity_score']