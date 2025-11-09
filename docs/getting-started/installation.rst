Installation
============
Environment Setup
-----------------

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/Lucas-Mueller/Rawls_v3.git
      cd Rawls_v3

2. **Create Virtual Environment**

   .. code-block:: bash

      # Create and activate virtual environment
      python -m venv .venv
      source .venv/bin/activate  # On macOS/Linux
      # OR on Windows:
      .venv\Scripts\activate

3. **Install Dependencies**

   .. code-block:: bash

      pip install -r requirements.txt

Core Dependencies
~~~~~~~~~~~~~~~~~

The system relies on these core packages:

- ``openai-agents`` - Multi-agent framework with model provider support
- ``python-dotenv`` - Environment variable management
- ``pydantic`` - Data validation and settings management
- ``PyYAML`` - Configuration file parsing

Analysis Dependencies
~~~~~~~~~~~~~~~~~~~~~

For data analysis and visualization:

- ``pandas`` - Data manipulation and analysis
- ``numpy`` - Numerical computing
- ``matplotlib`` - Basic plotting
- ``seaborn`` - Statistical visualization
- ``scipy`` - Scientific computing
- ``statsmodels`` - Statistical modeling
- ``plotly`` - Interactive visualization

API Key Configuration
---------------------

The system automatically retrieves API keys from your environment. You can set them up in several ways:

.env File 
~~~~~~~~~~~~~~~~~~~~~~~

Create a ``.env`` file in the project root:

.. code-block:: bash

   OPENAI_API_KEY=your-openai-key-here
   OPENROUTER_API_KEY=your-openrouter-key-here # Optional
   OLLAMA_BASE_URL=http://localhost:11434/v1
   OLLAMA_API_KEY=ollama



Next Steps
----------

With the system installed, head to the :doc:`quickstart` guide to run your first experiment!
