import re
from email import message_from_file
from email import message_from_string
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup

def extract_eml(file):
    """
    Extracts and processes content from a .eml file.

    Args:
        file_path (str): The path to the .eml file.

    Returns:
        dict: A dictionary containing the extracted fields.
    """
  
    msg = message_from_string(file)

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
            if content_type in ["text/plain", "text/html"]:
                charset = part.get_content_charset() or 'utf-8'  # Fallback to 'utf-8' if charset is None
                body_content = part.get_payload(decode=True)
                if body_content is not None:
                    body_content = body_content.decode(charset, errors="replace")
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
        charset = msg.get_content_charset() or 'utf-8'
        body_content = msg.get_payload(decode=True)
        if body_content is not None:
            body_content = body_content.decode(charset, errors="replace")
        
        # Remove all script and style tags
        soup = BeautifulSoup(body_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Preserve the clickable text and URLs together
        for a in soup.find_all("a", href=True):
            a.replace_with(f"{a.get_text()} ({a['href']})")

        body_plain = soup.get_text(separator=" ")

    # Remove extra spaces and newlines
    body_plain = re.sub(r'\s+', ' ', body_plain).strip()

    # Optional: parse date to datetime
    if date:
        try:
            parsed_date = parsedate_to_datetime(date)
        except (TypeError, ValueError) as e:
            parsed_date = None  # Assign a fallback if parsing fails
    else:
        parsed_date = None

    return {
        "from_email": from_email,
        "subject": subject,
        "date": parsed_date,
        "message_id": message_id,
        "return_path": return_path,
        "authentication_results": authentication_results,
        "sender_ip": sender_ip,
        "body_plain": body_plain
    }



# Example usage
