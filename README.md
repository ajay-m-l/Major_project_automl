# Conversational AutoML for Automated Data Analysis

## Overview

Conversational AutoML for Automated Data Analysis is an intelligent multi-agent system that enables users to perform end-to-end data analytics through natural language conversations. Instead of requiring expertise in programming, machine learning, or data visualization tools, users can interact with the system using simple English queries and receive analytical insights, visualizations, data cleaning operations, and machine learning predictions.

The system employs a supervisor-driven multi-agent architecture powered by LangGraph, where specialized agents collaborate to execute complex analytical workflows. A conversational interface allows users to upload datasets, ask questions, generate visualizations, clean data, and build machine learning models without writing code.

---

## Problem Statement

Traditional data analytics platforms require users to possess technical knowledge of programming languages, machine learning frameworks, statistical analysis, and visualization libraries. This creates a significant barrier for non-technical users and business stakeholders who need quick access to insights.

The objective of this project is to develop a conversational analytics platform that automates the complete data analysis lifecycle through natural language interactions.

---

## Key Features

### Multi-Agent Architecture

The system utilizes multiple specialized AI agents coordinated by a supervisor agent.

* Supervisor Agent

  * Interprets user intent
  * Routes tasks to appropriate agents
  * Coordinates workflow execution

* Data Analysis Agent

  * Performs exploratory data analysis
  * Generates descriptive statistics
  * Identifies trends and patterns

* Data Cleaning Agent

  * Handles missing values
  * Detects duplicates
  * Performs preprocessing operations

* Visualization Agent

  * Creates charts and graphs
  * Generates visual insights
  * Supports interactive analytics

* Machine Learning Agent

  * Trains predictive models
  * Evaluates model performance
  * Generates predictions

---

## System Architecture

```text
User
 │
 ▼
React Frontend
 │
 ▼
FastAPI Backend
 │
 ▼
Supervisor Agent (LangGraph)
 │
 ├── Analysis Agent
 ├── Cleaning Agent
 ├── Visualization Agent
 └── ML Agent
 │
 ▼
Dataset & Model Processing
 │
 ▼
Results / Insights / Visualizations
```

---

## Technology Stack

### Frontend

* React.js
* HTML5
* CSS3
* JavaScript

### Backend

* FastAPI
* Python

### AI and Agent Framework

* LangGraph
* LangChain
* Local LLM Integration (Ollama)

### Data Science Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

### Development Tools

* Git
* GitHub
* VS Code

---

## Project Structure

```text
project-root/
│
├── agents/
│   ├── executor.py
│   ├── supervisor_agent.py
│   └── react_agents.py
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
│
├── models/
│   └── .gitkeep
│
├── tools/
│   ├── analysis_tool.py
│   ├── cleaning_tool.py
│   ├── ml_tool.py
│   └── visualization_tool.py
│
├── utils/
│   ├── llm_wrapper.py
│   ├── memory.py
│   ├── ollama.py
│   └── schema.py
│
├── requirements.txt
├── README.md
└── sample.py
```

---

## Workflow

1. User uploads a dataset.
2. User enters a natural language query.
3. Supervisor Agent analyzes the request.
4. Appropriate specialized agents are selected.
5. Tools perform analysis, cleaning, visualization, or modeling.
6. Results are returned through the conversational interface.
7. Session memory maintains context for follow-up queries.

---

## Supported Operations

### Data Analysis

* Dataset overview
* Summary statistics
* Correlation analysis
* Feature inspection
* Trend identification

### Data Cleaning

* Missing value handling
* Duplicate removal
* Data transformation
* Outlier detection

### Visualization

* Bar Charts
* Line Charts
* Scatter Plots
* Histograms
* Correlation Heatmaps

### Machine Learning

* Classification
* Regression
* Model Evaluation
* Prediction Generation

---

## Installation

### Clone Repository

```bash
git clone https://github.com/ajay-m-l/Major_project_automl.git
cd Major_project_automl
```

### Create Virtual Environment

```bash
python -m venv myenv
```

### Activate Virtual Environment

#### Windows

```bash
myenv\Scripts\activate
```

#### Linux / macOS

```bash
source myenv/bin/activate
```

### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend Server

```bash
cd backend
uvicorn app:app --reload
```

Backend API will be available at:

```text
http://localhost:8000
```

### Install Frontend Dependencies

Open a new terminal:

```bash
cd frontend
npm install
```

### Run Frontend

```bash
npm start
```

Frontend application will be available at:

```text
http://localhost:3000
```


---

## Future Enhancements

* Multi-user authentication
* Cloud deployment
* Advanced AutoML pipelines
* Dashboard export functionality
* Real-time collaborative analytics
* Support for large-scale datasets
* Automated report generation

---

## Academic Relevance

This project demonstrates the integration of:

* Artificial Intelligence
* Multi-Agent Systems
* Natural Language Processing
* Automated Machine Learning
* Data Analytics
* Full Stack Development

The system serves as a practical implementation of conversational AI-driven analytics and showcases how intelligent agents can automate complex data science workflows.

---

## Authors

**Ajay M L**
Master of Computer Applications (MCA)
RV College of Engineering, Bengaluru

---

## License

This project is developed for academic and research purposes.
