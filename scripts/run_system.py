#!/usr/bin/env python3
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import os
import socket
import argparse

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def find_available_port(default_port: int = 8000) -> int:
    port = default_port
    while is_port_in_use(port) and port < default_port + 20:
        port += 1
    return port

def main():
    parser = argparse.ArgumentParser(description="Start ClinAI PACS Medical Diagnosis Platform")
    parser.add_argument("--port", type=int, default=None, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()

    default_port = int(os.environ.get("PORT", 8000))
    if args.port:
        port = args.port
    else:
        port = default_port if not is_port_in_use(default_port) else find_available_port(8005)

    print("=================================================================")
    print("   Starting AI-Based Medical Image Diagnosis System")
    print("   PACS Triage & Multi-Modal Clinical Intelligence Platform")
    print("=================================================================")
    print(f"Web UI Cockpit: http://localhost:{port}/")
    print(f"API Swagger Docs: http://localhost:{port}/docs")
    if port != default_port:
        print(f"Note: Port {default_port} was in use; automatically selected port {port}.")
    print("-----------------------------------------------------------------")
    uvicorn.run("backend.app.main:app", host=args.host, port=port, reload=False)

if __name__ == "__main__":
    main()
