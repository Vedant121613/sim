import requests

def download_pdf(url, save_path="temp.pdf"):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to 1download PDF: {response.status_code}")
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path
