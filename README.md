# ClipIt Backend

AI-powered video clip extraction service that analyzes videos to find the most interesting scenes based on user prompts.

## Features

- Upload video files and get AI-selected clips
- Face detection and motion analysis
- Google Gemini AI integration for intelligent scene selection
- RESTful API for easy integration
- Persistent storage for generated clips

## API Endpoints

- `POST /analyze` - Upload video and get extracted clip
- `GET /health` - Health check
- `POST /feedback` - Submit user feedback
- `GET /stats` - Get usage statistics

## Environment Variables

```bash
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8080
```

## Deployment

This backend is configured for deployment on fly.io:

1. Install fly CLI and login
2. Set environment variables: `flyctl secrets set GEMINI_API_KEY=your_key`
3. Deploy: `flyctl deploy`

## Local Development

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
python app.py
```