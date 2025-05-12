
# Omnica Python Web App (Flask & FastAPI)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-%E2%82%99.0+-green.svg)](https://flask.palletsprojects.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-blueviolet.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Maintenance](https://img.shields.io/maintenance/yes/2025)

This repository contains the source code for Omnica Python web application, which leverages both Flask and FastAPI frameworks.

## Overview

This application demonstrates how to integrate and utilize both Flask and FastAPI within a single project. It showcases potential use cases where you might choose one framework over the other for specific parts of your application.

**Key Features:**

* **Flask Integration:** Demonstrates basic routing and rendering with Flask.
* **FastAPI Integration:** Showcases API endpoint creation and data validation with FastAPI.
* **Clear Structure:** Well-organized project structure for easy understanding.
* **Basic Examples:** Provides simple examples for both frameworks.

## Project Structure
```
Omnica/
├── app.py           # Main application entry point
├── requirements.txt # Project dependencies
├── flask_module/    # Flask-specific code
│   ├── init.py
│   ├── routes.py
│   └── templates/
│       └── index.html
└── fastapi_module/  # FastAPI-specific code
├── init.py
└── main.py
```
## Getting Started

Follow these steps to get the application running on your local machine.

### Prerequisites

* **Python 3.8+** installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
* **pip** (Python package installer). It usually comes bundled with Python.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

**Option 1: Running Flask Separately**

1.  Navigate to the `flask_module` directory:
    ```bash
    cd flask_module
    ```

2.  Run the Flask application:
    ```bash
    python routes.py
    ```
    (or potentially `flask run` if configured)

3.  Open your web browser and go to `http://127.0.0.1:5000/` (or the address specified in your Flask configuration).

**Option 2: Running FastAPI Separately**

1.  Navigate to the `fastapi_module` directory:
    ```bash
    cd fastapi_module
    ```

2.  Run the FastAPI application using Uvicorn (an ASGI server):
    ```bash
    uvicorn main:app --reload
    ```
    (assuming your FastAPI application instance is named `app` in `main.py`)

3.  Open your web browser and go to `http://127.0.0.1:8000/docs` to see the automatically generated API documentation, or try your defined API endpoints directly.

**Option 3: Running Both (if integrated in `app.py`)**

* Follow the instructions within the `app.py` file or any accompanying documentation on how to run the combined application. This might involve running a specific script or using a tool like Gunicorn or Waitress to serve both applications.

## Usage

Provide examples of how to interact with your application. For instance:

* **Flask Example:** "Navigate to `/` in your browser to see the homepage rendered by Flask."
* **FastAPI Example:** "Send a GET request to `/api/items/{item_id}` to retrieve item details (replace `{item_id}` with an actual ID)."

## Contributing

If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b bugfix/your-bug-fix`.
3.  Make your changes and commit them: `git commit -m "Add your descriptive commit message"`.
4.  Push to the branch: `git push origin feature/your-feature-name` or `git push origin bugfix/your-bug-fix`.
5.  Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for more details.

## Acknowledgements

* [Flask](https://flask.palletsprojects.com/): The microframework for Python.
* [FastAPI](https://fastapi.tiangolo.com/): Modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.
* [Uvicorn](https://www.uvicorn.org/): An ASGI web server for Python.
* Any other libraries or resources you used.
