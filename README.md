# Omnica
# My First Flask API

This repository contains a simple "Hello, World!" API built using Python and the Flask microframework. It's a great starting point for anyone looking to learn how to create web APIs with Flask.

## Getting Started

These instructions will guide you on how to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* **Python 3.6+**: Make sure you have Python 3.6 or a later version installed on your system. You can check your Python version by running:
    ```bash
    python --version
    ```
* **pip**: Python package installer, which usually comes bundled with your Python installation. You'll need this to install Flask.

### Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd my-first-flask-api
    ```
    *(Replace `<repository_url>` with the actual URL of your GitHub repository)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
    * **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Flask:**
    ```bash
    pip install Flask
    ```

## Running the API

1.  **Navigate to the project directory** (if you're not already there):
    ```bash
    cd my-first-flask-api
    ```

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    *(Assuming your main application file is named `app.py`)*

    This will start the Flask development server. You'll likely see output similar to:
    ```
     * Serving Flask app 'app'
     * Debug mode: off
    ```
    *(Note: In development, debug mode is usually on. You might see `Debug mode: on`)*

3.  **Access the API endpoint:**
    Open your web browser or use a tool like `curl` or Postman to send a GET request to the following URL:
    ```
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    ```
    *(`127.0.0.1` is the local host, and `5000` is the default port Flask uses)*

    You should receive the response:
    ```
    Hello, World!
    ```

## API Endpoints

Currently, this API has one simple endpoint:

* **`/` (GET)**: Returns a "Hello, World!" message.

## Project Structure
