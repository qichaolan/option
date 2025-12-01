# LEAPS Ranker Web App

A production-ready web application for ranking LEAPS (Long-term Equity Anticipation Securities) options and simulating ROI.

## Features

- **LEAPS Ranking Table**: View ranked LEAPS call options for SPY and QQQ
- **ROI Simulator**: Calculate potential returns at different target prices
- **Responsive Design**: Works on desktop and mobile devices
- **Two Scoring Modes**:
  - High Probability: Favors contracts easier to reach profitability
  - High Convexity: Favors contracts with higher ROI potential

## Local Development

### Prerequisites

- Python 3.12+
- pip

### Setup

1. Install dependencies:
```bash
cd web
pip install -r requirements.txt
```

2. Run the development server:
```bash
cd web
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

3. Open http://localhost:8080 in your browser

## Cloud Run Deployment

### Prerequisites

- Google Cloud SDK (`gcloud`)
- Docker
- A Google Cloud project with billing enabled

### Deploy Steps

1. **Authenticate with Google Cloud**:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. **Enable required APIs**:
```bash
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
```

3. **Build and deploy** (from the repository root):
```bash
# Build the container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/leaps-ranker

# Deploy to Cloud Run
gcloud run deploy leaps-ranker \
    --image gcr.io/YOUR_PROJECT_ID/leaps-ranker \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 60 \
    --max-instances 3
```

4. **Access your app**: The deployment will output a URL like `https://leaps-ranker-xxxxx-uc.a.run.app`

### Alternative: Deploy with Docker Locally

```bash
# Build image (from repository root)
docker build -f web/Dockerfile -t leaps-ranker .

# Run container
docker run -p 8080:8080 leaps-ranker
```

## API Endpoints

### GET /api/tickers
Returns list of supported tickers with default target percentages.

### POST /api/leaps
Fetch ranked LEAPS for a ticker.

Request body:
```json
{
    "symbol": "QQQ",
    "target_pct": 0.5,
    "mode": "high_prob",
    "top_n": 20
}
```

### POST /api/roi-simulator
Simulate ROI at different target prices.

Request body:
```json
{
    "strike": 500,
    "premium": 25.50,
    "underlying_price": 480,
    "target_prices": [500, 550, 600, 650],
    "contract_size": 100
}
```

### GET /health
Health check endpoint for Cloud Run.

## Project Structure

```
web/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models
│   └── routes/
│       ├── __init__.py
│       └── leaps.py      # API routes
├── static/
│   ├── css/
│   │   └── style.css     # Responsive styles
│   └── js/
│       └── app.js        # Frontend logic
├── templates/
│   └── index.html        # Main page template
├── Dockerfile            # Cloud Run deployment
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Configuration

The app uses the config file at `config/leaps_ranker.yaml` for scoring weights and filtering parameters. Key settings:

- `scoring_modes`: Weights for ease vs ROI scoring
- `filtering.min_dte`: Minimum days to expiration (default: 365)
- `display.top_n`: Number of contracts to return (default: 20)

## Disclaimer

This tool is for educational purposes only. Not financial advice. Options trading involves significant risk.
