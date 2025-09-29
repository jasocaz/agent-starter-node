# Railway Deployment Guide

## Environment Variables

Set these in your Railway project:

```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
OPENAI_API_KEY=your_openai_api_key
PORT=3000
NODE_ENV=production
```

## Deployment Steps

1. **Connect to Railway:**
   - Go to [railway.app](https://railway.app)
   - Sign up/login with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `agent-starter-node` repository

2. **Set Environment Variables:**
   - Go to your project → Variables tab
   - Add all the environment variables listed above

3. **Deploy:**
   - Railway will automatically build and deploy
   - The service will be available at `https://your-project.railway.app`

## API Endpoints

- `GET /health` - Health check
- `POST /start` - Start captions for a room
- `POST /stop` - Stop captions for a room
- `GET /sessions` - List active sessions

## Usage

```bash
# Start captions
curl -X POST https://your-project.railway.app/start \
  -H "Content-Type: application/json" \
  -d '{"roomName": "test-room", "targetLanguage": "es"}'

# Stop captions
curl -X POST https://your-project.railway.app/stop \
  -H "Content-Type: application/json" \
  -d '{"roomName": "test-room"}'
```
