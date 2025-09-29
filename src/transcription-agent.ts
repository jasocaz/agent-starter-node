import {
  Room,
  RoomEvent,
  RemoteParticipant,
  Track,
  type RemoteTrack,
  TrackKind,
  AudioStream,
  type AudioFrame,
  combineAudioFrames,
} from '@livekit/rtc-node';
import OpenAI from 'openai';
import { toFile } from 'openai/uploads';

// Importing '@livekit/rtc-node' registers a WebRTC implementation for Node

interface TranscriptionAgentOptions {
  livekitUrl: string;
  token: string;
  roomName: string;
  targetLanguage: string;
}

export class TranscriptionAgent {
  private room: Room;
  private openai: OpenAI;
  private isRunning = false;
  private options: TranscriptionAgentOptions;

  constructor(options: TranscriptionAgentOptions) {
    this.options = options;
    this.room = new Room();
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  async start() {
    if (this.isRunning) {
      console.log('Agent already running');
      return;
    }

    try {
      // Connect to the room
      await this.room.connect(this.options.livekitUrl, this.options.token);
      console.log(`Connected to room: ${this.options.roomName}`);

      // Set up event listeners
      this.room.on(RoomEvent.TrackSubscribed, this.handleTrackSubscribed.bind(this));
      this.room.on(RoomEvent.ParticipantConnected, this.handleParticipantConnected.bind(this));
      this.room.on(RoomEvent.Disconnected, this.handleDisconnected.bind(this));

      this.isRunning = true;
      console.log('Transcription agent started');
    } catch (error) {
      console.error('Failed to start transcription agent:', error);
      throw error;
    }
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    try {
      await this.room.disconnect();
      this.isRunning = false;
      console.log('Transcription agent stopped');
    } catch (error) {
      console.error('Error stopping transcription agent:', error);
      throw error;
    }
  }

  private handleParticipantConnected(participant: RemoteParticipant) {
    console.log(`Participant connected: ${participant.identity}`);
  }

  private handleTrackSubscribed(track: RemoteTrack, publication: any, participant: RemoteParticipant) {
    if (track.kind === TrackKind.KIND_AUDIO) {
      console.log(`Audio track subscribed from: ${participant.identity}`);
      void this.startAudioCaptureLoop(track, participant);
    }
  }

  // No-op: using @livekit/rtc-node which provides a Node-compatible Room

  private async startAudioCaptureLoop(track: RemoteTrack, participant: RemoteParticipant) {
    // Create a Web-Streams reader over PCM16 audio frames
    const stream = new AudioStream(track, { sampleRate: 16000, numChannels: 1, frameSizeMs: 20 });
    const reader = stream.getReader();
    let buffer: AudioFrame[] = [];
    let collectedMs = 0;
    const targetMs = 3000;

    const vadThreshold = Number(process.env.VAD_THRESHOLD ?? 800); // amplitude threshold (0..32767)
    try {
      while (this.isRunning) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;
        const isMuted = (track as any)?.muted === true;
        if (isMuted) {
          buffer = [];
          collectedMs = 0;
          continue;
        }
        buffer.push(value);
        const frameMs = (value.samplesPerChannel / value.sampleRate) * 1000;
        collectedMs += frameMs;
        if (collectedMs >= targetMs) {
          const combined = combineAudioFrames(buffer);
          buffer = [];
          collectedMs = 0;
          if (this.computeRms(combined.data) < vadThreshold) {
            continue;
          }
          await this.processPcm16Frame(combined.data, combined.sampleRate, combined.channels, participant);
        }
      }
    } catch (err) {
      console.error('AudioStream loop error:', err);
    } finally {
      try {
        reader.releaseLock();
      } catch {}
    }
  }

  private async processPcm16Frame(data: Int16Array, sampleRate: number, channels: number, participant: RemoteParticipant) {
    try {
      const wav = this.encodeWav(data, sampleRate, channels);
      const transcription = await this.openai.audio.transcriptions.create({
        file: await toFile(wav, 'audio.wav', { type: 'audio/wav' }),
        model: process.env.OPENAI_STT_MODEL || 'gpt-4o-transcribe',
        language: 'en',
      });
      const transcribedText = (transcription as any)?.text?.trim?.();
      if (transcribedText) {
        await this.sendTranscriptionMessage(participant.identity, transcribedText);
        if (this.options.targetLanguage && this.options.targetLanguage !== 'en') {
          await this.translateAndSend(transcribedText, participant.identity);
        }
      }
    } catch (e) {
      console.error('processPcm16Frame error:', e);
    }
  }

  private encodeWav(pcm16: Int16Array, sampleRate: number, numChannels: number): Buffer {
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = pcm16.length * bytesPerSample;
    const buffer = Buffer.alloc(44 + dataSize);

    buffer.write('RIFF', 0);
    buffer.writeUInt32LE(36 + dataSize, 4);
    buffer.write('WAVE', 8);
    buffer.write('fmt ', 12);
    buffer.writeUInt32LE(16, 16); // PCM header size
    buffer.writeUInt16LE(1, 20); // PCM format
    buffer.writeUInt16LE(numChannels, 22);
    buffer.writeUInt32LE(sampleRate, 24);
    buffer.writeUInt32LE(byteRate, 28);
    buffer.writeUInt16LE(blockAlign, 32);
    buffer.writeUInt16LE(16, 34); // bits per sample
    buffer.write('data', 36);
    buffer.writeUInt32LE(dataSize, 40);
    Buffer.from(pcm16.buffer).copy(buffer, 44);
    return buffer;
  }

  private computeRms(pcm16: Int16Array): number {
    if (pcm16.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < pcm16.length; i++) {
      const sample = pcm16[i] ?? 0;
      sum += sample * sample;
    }
    const mean = sum / pcm16.length;
    return Math.sqrt(mean);
  }

  private async processAudioChunk(audioData: Float32Array[], participant: RemoteParticipant) {
    try {
      // Convert audio data to format suitable for OpenAI Whisper
      if (audioData.length === 0 || !audioData[0]) {
        console.log('No audio data to process');
        return;
      }
      
      const flatArray = new Float32Array(audioData.length * audioData[0].length);
      let offset = 0;
      for (const chunk of audioData) {
        flatArray.set(chunk, offset);
        offset += chunk.length;
      }

      // Convert to 16-bit PCM for Whisper
      const pcm16 = new Int16Array(flatArray.length);
      for (let i = 0; i < flatArray.length; i++) {
        const sample = flatArray[i] ?? 0;
        pcm16[i] = Math.max(-32768, Math.min(32767, sample * 32768));
      }

      // Create audio buffer for OpenAI
      const audioBuffer = Buffer.from(pcm16.buffer);

      // Transcribe with OpenAI Whisper (Node: use toFile instead of File)
      const transcription = await this.openai.audio.transcriptions.create({
        file: await toFile(audioBuffer, 'audio.wav', { type: 'audio/wav' }),
        model: 'whisper-1',
        language: 'en', // You can make this configurable
      });

      const transcribedText = (transcription as any)?.text?.trim?.();
      if (transcribedText) {
        console.log(`Transcription from ${participant.identity}: ${transcribedText}`);
        
        // Send transcription as chat message
        await this.sendTranscriptionMessage(participant.identity, transcribedText);
        
        // Translate if target language is different
        if (this.options.targetLanguage !== 'en') {
          await this.translateAndSend(transcribedText, participant.identity);
        }
      }
    } catch (error) {
      console.error('Error processing audio chunk:', error);
    }
  }

  private async sendTranscriptionMessage(speaker: string, text: string) {
    try {
      const message = `[Transcript] ${speaker}: ${text}`;
      await this.room.localParticipant?.publishData?.(
        new TextEncoder().encode(message),
        { reliable: true, topic: 'lk-chat' as any }
      );
      console.log(`Sent transcription: ${message}`);
    } catch (error) {
      console.error('Error sending transcription message:', error);
    }
  }

  private async translateAndSend(text: string, speaker: string) {
    try {
      const translation = await this.openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: `Translate the following text to ${this.options.targetLanguage}. Return only the translation, no additional text.`,
          },
          {
            role: 'user',
            content: text,
          },
        ],
        max_tokens: 100,
        temperature: 0.1,
      });

      const translatedText = translation.choices[0]?.message?.content;
      if (translatedText) {
        const message = `[Translation] ${speaker}: ${translatedText}`;
        await this.room.localParticipant?.publishData?.(
          new TextEncoder().encode(message),
          { reliable: true, topic: 'lk-chat' as any }
        );
        console.log(`Sent translation: [Translation] ${speaker}: ${translatedText}`);
      }
    } catch (error) {
      console.error('Error translating text:', error);
    }
  }

  private handleDisconnected() {
    console.log('Disconnected from room');
    this.isRunning = false;
  }
}
