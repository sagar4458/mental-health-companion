# Solace — Roadmap

## Current Version (June 2026)

- [x] VADER sentiment analysis — compound score + intensity classification
- [x] Keyword-based multi-emotion scoring across 5 mood profiles
- [x] Google Gemini 1.5 Flash + Groq Llama 3.1 dual LLM setup
- [x] Full Gemini model fallback chain (5 models)
- [x] Crisis detection — 14 keywords + VADER threshold → instant helpline response
- [x] CBT + mindfulness + person-centred counselling system prompt
- [x] Adaptive background theme — JS RGB interpolation per detected emotion
- [x] Emotional Analysis Panel — emotion bars, sentiment waveform, session arc
- [x] Session Intelligence Bar — exchanges, emotion journey, stress/energy
- [x] Plutchik emotion wheel with live pointer
- [x] Typewriter text reveal — speed varies by emotion
- [x] Breathing synchronization overlay — 4-4-6-2 box breathing
- [x] Memory timeline — colored journey dots per exchange
- [x] Particle weather system — canvas particles match emotional state
- [x] Crisis UI — deep crimson theme, full-screen alert on detection

---

## Planned Improvements

### NLP & Emotion Intelligence
- [ ] Replace keyword matching with a fine-tuned transformer — `distilbert-base-uncased` or `roberta-base` fine-tuned on GoEmotions dataset (58k labeled Reddit comments, 27 emotion classes)
- [ ] Multi-label emotion classification — detect mixed emotions simultaneously (e.g. anxious + hopeful at 60/40 split) instead of single primary mood
- [ ] Sarcasm and negation handling — current VADER misclassifies "I'm totally fine" as positive
- [ ] Named Entity Recognition — detect people, places, events the user mentions and reference them in future responses
- [ ] Emotion intensity regression — continuous 0–1 scale instead of low/medium/high buckets
- [ ] Hugging Face `pipeline("text-classification")` integration for emotion scoring as a drop-in upgrade

### RAG — Memory and Context
- [ ] Long-term memory with ChromaDB — store user session summaries as vector embeddings
- [ ] Retrieve past relevant sessions when user returns — "Last time you mentioned your exams were causing anxiety..."
- [ ] RAG-powered coping strategy retrieval — index CBT workbooks, mindfulness guides and match strategies to the specific emotion detected
- [ ] Session summarization with Gemini — auto-generate a 3-line session summary after each conversation ends
- [ ] Nomic-embed-text or `sentence-transformers/all-MiniLM-L6-v2` for local embedding generation
- [ ] User preference memory — remember preferred coping strategies and communication style

### Vision-Based Emotion Analysis
- [ ] Facial expression analysis via webcam — integrate MediaPipe Face Mesh + a fine-tuned CNN on FER-2013 dataset
- [ ] Fuse text emotion + facial emotion for a combined confidence score
- [ ] Voice tone analysis — detect stress, sadness and anxiety from audio input using `librosa` feature extraction
- [ ] Multimodal emotion input — combine text, face and voice signals into a unified emotion vector
- [ ] Real-time facial action unit (AU) detection — track AU4 (brow lowerer), AU12 (lip corner puller), AU17 (chin raiser)

### Advanced Sentiment Analysis
- [ ] Aspect-based sentiment analysis — detect sentiment toward specific topics (work, relationships, self-image) independently
- [ ] Temporal sentiment tracking — plot compound score over the full session as a line graph inside the analysis panel
- [ ] Longitudinal mood analysis — track user emotional patterns across multiple sessions over weeks
- [ ] Depression and anxiety screening proxies — PHQ-9 and GAD-7 style conversational assessment built into extended sessions
- [ ] Emotion cause detection — identify triggers ("my boss", "the exam", "being alone") using dependency parsing

### LLM and Response Quality
- [ ] Fine-tune Llama 3.1 8B on mental health conversation datasets — EmpatheticDialogues (25k empathetic conversations) and Counsel Chat
- [ ] Safety classifier layer — secondary model that checks every response for harmful content before display
- [ ] Response diversity — track and avoid repeating the same reflection or question in the same session
- [ ] Structured response templates per emotion type — different flow for crisis vs sadness vs anxiety
- [ ] Ollama local deployment option — run Llama 3.2 3B fully offline for privacy-first use

### UI and Experience
- [ ] Breathing exercise library — 4-7-8 breathing, diaphragmatic breathing, coherent breathing — selectable per session
- [ ] Mood check-in on session start — one-tap emotion selector before first message
- [ ] Session replay — scroll back through emotion arc with timestamps
- [ ] Exportable session report — PDF summary of conversation, emotion arc and coping strategies used
- [ ] Dark / light / high-contrast theme toggle
- [ ] Mobile-responsive layout

### Deployment
- [ ] Deploy on Render or Railway with persistent sessions
- [ ] Public URL for portfolio and GitHub demo link
- [ ] Environment variable management for production secrets
- [ ] Rate limiting and session timeout handling for public deployment
