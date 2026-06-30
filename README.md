# JEE 2025 College Advisor 🎓

An intelligent, AI-powered portal to help JEE students find the best college + branch combinations based on their rank, home state, gender, and interests.

## Features

- ✅ Filter colleges by JEE Mains OR Advanced rank
- ✅ Smart quota logic (HS/OS for NITs, AI for IITs)
- ✅ IITs shown ONLY if JEE Advanced rank provided
- ✅ Interest-based branch scoring (coding, research, MBA, core engg)
- ✅ Chance label per option (Safe / Good / Moderate / Reach)
- ✅ Click any college → detailed modal with official website + AI insights
- ✅ AI chat advisor (Groq / LLaMA-3.3-70B) with student context
- ✅ Quick question shortcuts in chat
- ✅ Filter/search results by college type, chance, keyword

## Setup

### 1. Clone and install
```bash
git clone <repo>
cd jee_portal
pip install -r requirements.txt
```

### 2. Get free Groq API key
Visit https://console.groq.com → sign up → create API key (free tier available)

### 3. Set environment variable
```bash
# Linux / macOS
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"

# Windows
set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 4. Run the server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open in browser
```
http://localhost:8000
```

## Data Files (place in `data/` folder)
- `JEE_2025_Cutoffs.xlsx` — 2410 rows of institute/program cutoffs
- `JEE_2025_ALL_128_COLLEGES.xlsx` — 128 colleges with official websites

## Architecture

```
main.py          ← FastAPI app, filtering logic, Groq AI integration
templates/
  index.html     ← Full single-page frontend (vanilla JS, no framework)
data/
  *.xlsx         ← JEE 2025 data
static/          ← Static assets (if any)
requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Main portal UI |
| POST | /api/get-options | Get filtered college options |
| POST | /api/college-detail | Get college details + AI insights |
| POST | /api/chat | AI advisor chat |
| GET | /api/states | List of states |
| GET | /api/health | Health check |

## Quota Logic

| Institute Type | Quota Applied |
|---------------|---------------|
| IIT | AI (All India) — requires JEE Advanced |
| NIT (home state) | HS (Home State) |
| NIT (other state) | OS (Other State) |
| IIIT / Others | AI |

## Rank Window

Results are shown for programs where:
- `Closing Rank >= student_rank × 0.70`
- `Opening Rank <= student_rank × 1.50`

This ensures the student sees realistic options — not too easy, not too far out of reach.

## License
MIT — open source, free to use and modify.
