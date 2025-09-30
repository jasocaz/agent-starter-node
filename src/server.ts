import express from 'express';
import cors from 'cors';
import { AccessToken, RoomServiceClient } from 'livekit-server-sdk';
import { TranscriptionAgent } from './transcription-agent.js';

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Store active agent sessions
const activeSessions = new Map<string, TranscriptionAgent>();

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start captions for a room
app.post('/start', async (req, res) => {
  try {
    const { roomName, targetLanguage, sttLanguage } = req.body;
    
    if (!roomName) {
      return res.status(400).json({ error: 'Missing roomName' });
    }

    // Check if already running
    if (activeSessions.has(roomName)) {
      return res.json({ message: 'Captions already running for this room' });
    }

    // Create agent token
    const livekitUrl = process.env.LIVEKIT_URL;
    const livekitApiKey = process.env.LIVEKIT_API_KEY;
    const livekitApiSecret = process.env.LIVEKIT_API_SECRET;

    if (!livekitUrl || !livekitApiKey || !livekitApiSecret) {
      return res.status(500).json({ error: 'LiveKit credentials not configured' });
    }

    const at = new AccessToken(livekitApiKey, livekitApiSecret, {
      identity: `captions-agent-${Date.now()}`,
      name: 'Captions Agent',
    });

    at.addGrant({
      room: roomName,
      roomJoin: true,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    });

    const token = await at.toJwt();

    // Create and start transcription agent
    const agent = new TranscriptionAgent({
      livekitUrl,
      token,
      roomName,
      targetLanguage: targetLanguage || 'en',
      sttLanguage: sttLanguage || undefined,
    });

    await agent.start();
    activeSessions.set(roomName, agent);

    console.log(`Started captions agent for room: ${roomName} (targetLanguage=${targetLanguage || 'en'}, sttLanguage=${sttLanguage || 'auto'})`);
    return res.json({ message: 'Captions started successfully', roomName });
  } catch (error) {
    console.error('Error starting captions:', error);
    return res.status(500).json({ error: 'Failed to start captions' });
  }
});

// Stop captions for a room
app.post('/stop', async (req, res) => {
  try {
    const { roomName } = req.body;
    
    if (!roomName) {
      return res.status(400).json({ error: 'Missing roomName' });
    }

    const agent = activeSessions.get(roomName);
    if (!agent) {
      return res.json({ message: 'No active captions for this room' });
    }

    await agent.stop();
    activeSessions.delete(roomName);

    console.log(`Stopped captions agent for room: ${roomName}`);
    return res.json({ message: 'Captions stopped successfully', roomName });
  } catch (error) {
    console.error('Error stopping captions:', error);
    return res.status(500).json({ error: 'Failed to stop captions' });
  }
});

// Auto-cleanup: if no non-agent participants remain, stop the agent after a short delay
setInterval(async () => {
  try {
    for (const [roomName, agent] of activeSessions) {
      // We cannot inspect Room participants from here; rely on LK RoomService if configured later.
      // Placeholder hook for future enhancement.
    }
  } catch {}
}, 60000);

// List active sessions
app.get('/sessions', (req, res) => {
  const sessions = Array.from(activeSessions.keys());
  res.json({ activeRooms: sessions });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Shutting down gracefully...');
  for (const [roomName, agent] of activeSessions) {
    try {
      await agent.stop();
      console.log(`Stopped agent for room: ${roomName}`);
    } catch (error) {
      console.error(`Error stopping agent for room ${roomName}:`, error);
    }
  }
  process.exit(0);
});

app.listen(port, () => {
  console.log(`Captions agent server running on port ${port}`);
});
