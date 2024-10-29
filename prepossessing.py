import re
from email import message_from_string
from email.policy import default
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
nltk.download('wordnet')
nltk.download('stopwords')




def extract_email_info(raw_email):

    
    # Parse the email content
    email_message = message_from_string(raw_email, policy=default)
    
    # Extract metadata with default values if fields are missing
    email_info = {
        "subject": email_message["Subject"] if email_message["Subject"] else "No Subject",
        "from": email_message["From"] if email_message["From"] else "Unknown Sender",
        "to": email_message["To"] if email_message["To"] else "Unknown Recipient",
        "date": email_message["Date"] if email_message["Date"] else "No Date",
    }
    
    # Extract IP address from received headers
    received_headers = email_message.get_all("Received") or []
    ip_addresses = []
    for header in received_headers:
        # Use regex to find all IP addresses in the received headers
        ip_matches = re.findall(r'[0-9]+(?:\.[0-9]+){3}', header)
        ip_addresses.extend(ip_matches)
    email_info["ip_addresses"] = ip_addresses if ip_addresses else ["No IP addresses found"]
    
    # Extract email body content
    body = ""
    if email_message.is_multipart():
        for part in email_message.iter_parts():
            # Check if part is text/plain or text/html
            if part.get_content_type() in ["text/plain", "text/html"]:
                body += part.get_content() + "\n"
    else:
        body = email_message.get_content()
    
    content,url = separate_text_and_urls(body)
    email_info["body"] = content.strip() if content else "No Body Content"
    email_info["url"] = url if url else "None"
    
    return email_info


def separate_text_and_urls(email_content):
    # Enhanced regex pattern to capture complex domains and paths
    # url_pattern = re.compile(
    #     r'((https?://|www\.)?[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+(/[^\s]*)?)'
    # )
    regex = r"(?<![@\w:])(?:https?://|ftp://|www\.)[a-zA-Z0-9.-]+(?:[/a-zA-Z0-9.-]*)[^\s<>,'\"]\b/?|(?<![@\w:])[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:[/a-zA-Z0-9.-]*)[^\s<>,'\"]\b/?|(?:\d{1,3}\.){3}\d{1,3}(?:[/a-zA-Z0-9.-]*)[^\s<>,'\"]\b/?|http://\[[0-9a-fA-F:]+\](?:[/a-zA-Z0-9.-]*)[^\s<>,'\"]\b/?|ftp:[a-zA-Z0-9.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|http://\[[0-9a-fA-F:]+\](?:\:[0-9]+)?[/a-zA-Z0-9.-]*"

    # Find all URLs in the email content
    # urls = url_pattern.findall(email_content)
    urls =re.findall(regex,email_content)
    
    # processed_urls = []
    # for url in urls:
    #     full_url = url[0]
    #     processed_urls.append(full_url)

    processed_urls = urls  # Each match is already a full URL
    
    # Remove URLs from the original content
    text_content = re.sub(regex, '', email_content).strip()
    
    # # Remove URLs from the original content
    # text_content = url_pattern.sub('', email_content).strip()
    
    return text_content, processed_urls


def tokenize_text(body):
    tokens = body.split()
    return tokens

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

# Function to clean and remove stopwords/unwanted characters
def remove_stopwords_and_chars(lemmas):
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords and empty strings after cleaning
    lemmas = [token for token in lemmas if token.lower() not in stop_words and token]

    # Convert list of lemmas back to a string without commas
    return ' '.join(lemmas)

def prepossessing_content(raw_content):
    email_data = extract_email_info(raw_content)
    email_data["body"] = tokenize_text(email_data["body"])
    email_data["body"] = lemmatize_text(email_data["body"])
    email_data["body"] = remove_stopwords_and_chars(email_data["body"])

    return email_data

raw_email = """
Hello,

Please visit our website at https://example.com for more information.
If you have any questions, contact us at ple.org/help.

Best regards,
Example Team
"""
# email_data = extract_email_info(raw_email)

# # # print("Before: ")
# # # print(email_data)
# # print(f"*"*100 +"/n")
# # print("After: ")
# email_data["body"] = tokenize_text(email_data["body"])
# print(email_data["body"])
# email_data["body"] = lemmatize_text(email_data["body"])
# print(email_data["body"])
# email_data["body"] = remove_stopwords_and_chars(email_data["body"])
# print(email_data["body"])
# # print(email_data)

# result = prepossessing_content(raw_email)
# print(result)
