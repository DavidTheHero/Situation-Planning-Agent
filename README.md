# Situation-Planning Agent

A hybrid AI-algorithmic system for solving complex rail logistics planning problems using GPT-4o with automatic error correction and fallback mechanisms.

## 🚀 Features

- **AI-Powered Planning**: Leverages GPT-4o for intelligent plan generation
- **Automatic Error Correction**: 7 patches fix common planning mistakes
- **Strategic Analysis**: Problem pattern detection and targeted hints
- **Algorithmic Fallback**: Graph-based solver as backup when AI fails
- **83.3% Success Rate**: Solves 15/18 complex rail planning problems with AI alone

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Situation-Planning-Agent.git
cd Situation-Planning-Agent
```

2. Install dependencies:
```bash
pip install dspy python-dotenv
```

3. Configure your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## 🎯 Quick Start

### Test a single problem:
```bash
python minimal_test_patched.py --ids "1-A"
```

### Test multiple problems:
```bash
python minimal_test_patched.py --ids "1-A,2-B,3-C"
```

### Run with different modes:
```bash
# Baseline mode (no enhancements)
python minimal_test_patched.py --baseline --ids "1-A"

# Without fallback
python minimal_test_patched.py --no-fallback --ids "1-A"

# Without patches
python minimal_test_patched.py --no-patches --ids "1-A"
```

## 📊 Performance

| Mode | Success Rate | Problems Solved |
|------|--------------|-----------------|
| **Baseline** | 33.3% | 6/18 |
| **Enhanced** | 83.3% | 15/18 |
| **Enhanced + Fallback** | 88.9% | 16/18 |

*Results verified: August 18, 2024*

## 🏗️ System Architecture

### Core Components

- **`minimal_test_patched.py`** - Main enhanced system with patches and strategic analysis
- **`minimal_test_option1.py`** - Baseline AI planner
- **`algorithmic_solver.py`** - Dynamic fallback solver using graph algorithms
- **`optimized_solver.py`** - Time-optimized algorithmic solver
- **`improved_algorithmic_solver.py`** - Enhanced algorithmic solver
- **`fallback_solutions.py`** - Hardcoded solutions for specific problems
- **`checker.py`** - Plan validation system
- **`system_map.py`** - Rail network topology
- **`problem_parser.py`** - Problem parsing utilities

### Data Files

- **`all_plans_description.json`** - Problem definitions with goals and constraints
- **`cisd_cleaned.json`** - Natural language problem descriptions

## 🔧 Configuration Options

| Flag | Description |
|------|-------------|
| `--baseline` | Use original planner without enhancements |
| `--no-patches` | Disable automatic error corrections |
| `--no-fallback` | Disable algorithmic fallback |
| `--no-strategic` | Disable strategic analysis |
| `--ids "1-A,2-B"` | Specify problems to test |

## 📈 How It Works

### 1. Problem Analysis
System analyzes the problem to detect patterns:
- Capacity constraints
- Multi-destination deliveries
- Conversion requirements
- Resource availability

### 2. Strategic Planning
Enhanced prompts guide AI with problem-specific hints based on detected patterns.

### 3. AI Generation
GPT-4o generates initial plan using enhanced prompts with strategic guidance.

### 4. Automatic Patches
Seven patches fix common errors:
1. **Engine location enforcement** - Ensures valid starting positions
2. **Illegal travel expansion** - Converts invalid routes to valid paths
3. **Capacity violation checking** - Enforces 3-unit maximum per engine
4. **CONVERT quantity addition** - Adds missing parameters
5. **Missing delivery completion** - Ensures cargo reaches destination
6. **Excess step trimming** - Removes redundant operations
7. **Multi-destination quantity** - Handles DETACH operations

### 5. Validation
Plan is checked against goals and constraints using the checker system.

### 6. Fallback (if needed)
Algorithmic solver attempts problem if AI solution fails validation.

## 🎯 Problem Categories

- **Simple** (5 problems): Basic single/dual cargo deliveries - 100% success
- **Medium** (6 problems): Multi-destination, mixed cargo - 100% success with enhancements
- **Complex** (7 problems): Capacity challenges, conversions - 57% success with AI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [DSPy](https://github.com/stanfordnlp/dspy) framework
- Powered by OpenAI GPT-4o
- Rail planning domain from logistics research

---

*Version 1.0 | August 2024*