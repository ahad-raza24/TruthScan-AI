# 🔍 TruthScan - AI-Powered Fake News Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![RoBERTa](https://img.shields.io/badge/Model-RoBERTa-orange.svg)](https://huggingface.co/roberta-base)
[![DeBERTa](https://img.shields.io/badge/Model-DeBERTa-green.svg)](https://huggingface.co/microsoft/deberta-v3-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TruthScan** is an advanced, multi-agent AI system designed to detect fake news and misinformation using state-of-the-art transformer models (RoBERTa & DeBERTa), knowledge graph verification, and comprehensive bias analysis techniques.

📖 **[Technical Documentation](./TECHNICAL_DOCUMENTATION.md)** | 🚀 **[Quick Start](#-quick-start)** | 🔬 **[Research Paper](./research/)** | 🤝 **[Contributing](#-contributing)**

## 🌟 Features

### 🤖 **Multi-Agent Architecture**
- **News Retrieval Agent**: Fetches relevant articles from NewsAPI, MediaStack, and GNews
- **Fact-Checking Agent**: Verifies claims using RoBERTa/DeBERTa models + knowledge graphs
- **Bias Analysis Agent**: Detects political, emotional, and source bias using MPQA lexicons
- **LLM Decision Agent**: Makes final verdicts using DeepSeek R1 via OpenRouter

### 🎯 **Processing Modes**
- **🚀 FAST**: Quick verification for high-volume processing
- **⚖️ BALANCED**: Optimal balance of speed and accuracy (default)
- **🎯 ACCURATE**: Maximum accuracy for critical fact-checking

### 🔧 **Advanced Features**
- **🧠 Ensemble ML Models**: RoBERTa + DeBERTa with custom neural network fusion
- **📊 Knowledge Graph Integration**: Real-time fact verification against structured knowledge
- **🎯 Multi-Source News Retrieval**: NewsAPI, GNews, MediaStack with intelligent deduplication
- **⚖️ Comprehensive Bias Detection**: Political, emotional, and source credibility analysis
- **🌐 Web Interface**: User-friendly dashboard with real-time verification
- **🔌 REST API**: Full programmatic access with detailed response schemas
- **📁 File Upload Support**: PDF, DOCX, and TXT processing with content extraction

### 🛡️ **Enterprise Ready**
- **Type-Safe Configuration**: Full type hints and validation
- **Environment Variable Support**: Secure API key management
- **Comprehensive Logging**: Detailed system monitoring
- **Error Handling**: Graceful degradation and recovery
- **Rate Limiting**: Built-in protection against abuse

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/truthscan.git
cd truthscan

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Build and run with Docker Compose
make quick-start

# Or manually:
docker-compose up -d

# Access the application
open http://localhost:5006
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/truthscan.git
cd truthscan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
python app.py
```

## 📋 Prerequisites

### Required API Keys
- **OpenRouter API Key**: For DeepSeek R1 LLM decision making ([Get API Key](https://openrouter.ai/))
- **News API Key**: For news article retrieval ([Get API Key](https://newsapi.org/register))
- **Knowledge Graph API Key**: For Wikidata/DBpedia fact verification (optional)
- **MediaStack API Key**: Additional news source (optional, [Get API Key](https://mediastack.com/))
- **GNews API Key**: Google News aggregation (optional, [Get API Key](https://gnews.io/))

### System Requirements
- **Python**: 3.10 or higher (required for transformer models)
- **Memory**: 4GB RAM minimum (8GB recommended for ensemble models)
- **Storage**: 2GB for models and dependencies
- **Network**: Internet connection for API access and model downloads

## 🤖 AI Models & Technology

### Transformer Models
- **RoBERTa-base**: 125M parameters, optimized for social media and informal text
- **DeBERTa-v3-small**: 140M parameters with disentangled attention mechanism
- **Ensemble Architecture**: Custom neural network fusion combining both models

### Training Details
- **Dataset Support**: Compatible with LIAR, ISOT, FakeNewsNet, WELFake datasets
- **Features**: 16-dimensional engineered features + TF-IDF (5000 dims) + embeddings (384 dims)
- **Performance**: 87-93% accuracy depending on processing mode
- **Training**: PyTorch with AdamW optimizer, mixed precision, early stopping

> 📖 **For detailed technical information**, see [Technical Documentation](./TECHNICAL_DOCUMENTATION.md)

## ⚙️ Configuration

TruthScan uses a unified configuration system with three processing modes:

```python
from system_configuration import SystemConfiguration, ProcessingMode

# Choose your processing mode
config = SystemConfiguration(mode=ProcessingMode.BALANCED)

# Customize settings
config.knowledge_graph.timeout_seconds = 10.0
config.news_retrieval.max_articles_per_query = 15
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required API Keys
OPENROUTER_API_KEY=your_openrouter_key_here
NEWS_API_KEY=your_news_api_key_here

# Optional API Keys
KNOWLEDGE_GRAPH_API_KEY=your_kg_api_key_here
MEDIASTACK_API_KEY=your_mediastack_key_here
GNEWS_API_KEY=your_gnews_key_here

# Configuration Overrides
TRUTHSCAN_MODE=balanced
KG_ENABLED=true
KG_TIMEOUT=6.0
MAX_ARTICLES=10
LOG_LEVEL=INFO
```

## 🖥️ Usage

### Web Interface

1. **Navigate to** `http://localhost:5006`
2. **Enter a claim** to verify in the text box
3. **Select processing mode** (Fast/Balanced/Accurate)
4. **Click "Verify Claim"** to start analysis
5. **View results** including verdict, confidence score, and evidence

### API Usage

#### Verify a Claim

```bash
curl -X POST http://localhost:5006/detect \
  -F "claim=AI will replace all human jobs by 2030" \
  -F "source=tech-news.com"
```

#### Health Check

```bash
curl http://localhost:5006/api/health
```

#### Get Configuration

```bash
curl http://localhost:5006/api/config
```

### Python Integration

```python
from system_configuration import SystemConfiguration, ProcessingMode
from fakenews_multi_agent import FakeNewsDetectionSystem

# Initialize system
config = SystemConfiguration(mode=ProcessingMode.ACCURATE)
system = FakeNewsDetectionSystem(config=config)

# Verify a claim
result = system.verify_claim(
    claim="Climate change is a hoax",
    source="example.com"
)

print(f"Verdict: {result['decision']['verdict']}")
print(f"Confidence: {result['decision']['trust_score']}")
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TruthScan System                         │
├─────────────────────────────────────────────────────────────┤
│  Web Interface (Flask)  │  REST API  │  Python SDK         │
├─────────────────────────────────────────────────────────────┤
│                Multi-Agent Orchestrator                     │
├─────────────────────────────────────────────────────────────┤
│ News Retrieval │ Fact Checking │ Bias Analysis │ LLM Decision│
│     Agent      │     Agent     │     Agent     │    Agent    │
├─────────────────────────────────────────────────────────────┤
│  NewsAPI  │ MediaStack │ Knowledge │  NLP Models │ OpenRouter │
│  GNews    │   APIs     │  Graphs   │  Transformers│    LLM    │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Processing Modes Comparison

| Feature | FAST | BALANCED | ACCURATE |
|---------|------|----------|----------|
| Knowledge Graph | ❌ | ✅ (6s) | ✅ (12s) |
| Articles Retrieved | 5 | 8 | 15 |
| Content Extraction | ❌ | ❌ | ✅ |
| Advanced NLP | ❌ | ✅ | ✅ |
| LLM Timeout | 15s | 25s | 45s |
| **Use Case** | Demos, High-volume | Production | Critical Analysis |

## 🔌 API Reference

### Endpoints

#### `POST /detect`
Verify a news claim

**Parameters:**
- `claim` (string, required): The claim to verify
- `source` (string, optional): Source of the claim

**Response:**
```json
{
  "success": true,
  "verification_id": "20250127123456",
  "claim": "Example claim",
  "final_result": {
    "verdict": "likely_false",
    "trust_score": 0.25,
    "confidence": 85,
    "explanation": "Detailed explanation...",
    "contributing_factors": ["factor1", "factor2"]
  }
}
```

#### `POST /verify_article`
Verify a full news article

**Parameters:**
- `title` (string): Article title
- `text` (string): Article content
- `source` (string, optional): Article source

#### `POST /upload`
Upload and verify document content

**Parameters:**
- `docfile` (file): PDF, DOCX, or TXT file
- `claim` (string, optional): Additional claim text

#### `GET /api/health`
System health check

#### `GET /api/config`
Get current system configuration

## 🐳 Docker Deployment

### Quick Start with Make

```bash
# Set up and start everything
make quick-start

# View logs
make logs

# Stop services
make down
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Docker directly

```bash
# Build image
make build

# Run container
make run
```

### Available Make Commands

```bash
make help          # Show all available commands
make setup         # Set up development environment
make dev           # Start development server locally
make build         # Build Docker image
make up            # Start with docker-compose
make down          # Stop services
make logs          # Show application logs
make clean         # Clean up Docker resources
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/truthscan.git
cd truthscan

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=.

# Run specific test file
python -m pytest tests/test_fact_checker.py
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📁 Project Structure

```
truthscan/
├── app.py                      # Flask web application
├── system_configuration.py    # Configuration management
├── fakenews_multi_agent.py    # Multi-agent orchestrator
├── agents/
│   ├── base_agent.py          # Base agent class
│   ├── news_retrieval_agent.py # News fetching
│   ├── fact_check_agent.py    # Fact verification
│   ├── bias_analysis_agent.py # Bias detection
│   └── llm_agent.py           # LLM decision making
├── templates/                  # HTML templates
├── static/                     # CSS, JS, images
├── lexicons/                   # Bias detection lexicons
├── models/                     # ML models (if any)
├── logs/                       # Application logs
├── results/                    # Verification results
├── docker/
│   ├── Dockerfile             # Docker image definition
│   └── docker-compose.yml     # Multi-container setup
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `python -m pytest`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy** for natural language processing
- **Transformers** for state-of-the-art NLP models
- **Flask** for the web framework
- **OpenRouter** for LLM access
- **NewsAPI** for news data access

## 📞 Support

- **Documentation**: [Full Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/truthscan/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/truthscan/discussions)
- **Email**: support@truthscan.ai

## 🔮 Roadmap

- [ ] **Real-time fact-checking** browser extension
- [ ] **Mobile app** for iOS and Android
- [ ] **Advanced ML models** for better accuracy
- [ ] **Multi-language support** for global use
- [ ] **Blockchain integration** for transparency
- [ ] **Community fact-checking** features

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[Report Bug](https://github.com/yourusername/truthscan/issues) • [Request Feature](https://github.com/yourusername/truthscan/issues) • [Documentation](docs/)

</div> 