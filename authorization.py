# authorization.py

API_KEY = "supersecretkey12"  # Replace with your actual secret token

def validate_token(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ")[1]
    return token == API_KEY
