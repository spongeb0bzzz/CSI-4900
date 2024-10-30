import re
from email import message_from_file
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup

# Load .eml file
file_path = "test.eml"  # Replace with the path to your .eml file
with open(file_path, "r", encoding="utf-8") as file:
    msg = message_from_file(file)

# Extract fields
from_email = msg.get("From")
subject = msg.get("Subject")
date = msg.get("Date")
message_id = msg.get("Message-ID")
return_path = msg.get("Return-Path")
authentication_results = msg.get("ARC-Authentication-Results")

# Get sender IP from Received headers
received_headers = msg.get_all("Received")
sender_ip = None
for header in received_headers:
    ip_match = re.search(r'\[(\d{1,3}(?:\.\d{1,3}){3})\]', header)
    if ip_match:
        sender_ip = ip_match.group(1)
        break  # Use the first matched IP as the originating IP

# Extract and clean body (plain text, retaining URLs and clickable text)
body_plain = ""
if msg.is_multipart():
    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain" or content_type == "text/html":
            body_content = part.get_payload(decode=True).decode(part.get_content_charset(), errors="replace")
            # Parse HTML and keep URLs and content within tags
            soup = BeautifulSoup(body_content, "html.parser")
            
            # Remove all script and style tags
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()  # Remove scripts and CSS

            # Preserve the clickable text and URLs together
            for a in soup.find_all("a", href=True):
                a.replace_with(f"{a.get_text()} ({a['href']})")
            
            body_plain = soup.get_text(separator=" ")  # Extract text with single-space separator
            break  # Stop at the first text-based part
else:
    body_content = msg.get_payload(decode=True).decode(msg.get_content_charset(), errors="replace")
    soup = BeautifulSoup(body_content, "html.parser")
    
    # Remove all script and style tags
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Preserve the clickable text and URLs together
    for a in soup.find_all("a", href=True):
        a.replace_with(f"{a.get_text()} ({a['href']})")

    body_plain = soup.get_text(separator=" ")

# Remove extra spaces and newlines
body_plain = re.sub(r'\s+', ' ', body_plain).strip()

# Optional: parse date to datetime
parsed_date = parsedate_to_datetime(date) if date else None

# Output results
print(f"From: {from_email}\n")
print(f"Sender IP: {sender_ip}\n")
print(f"Subject: {subject}\n")
print(f"Date: {parsed_date}\n")
print(f"Message-ID: {message_id}\n")
print(f"Return-Path: {return_path}\n")
print(f"Authentication Results: {authentication_results}\n")
print(f"Body (plain text, no HTML tags or CSS, URLs and clickable text intact): {body_plain}\n")
