@echo off
:: This script installs all the necessary dependencies for the Fake News Detector project.
:: It can be run by double-clicking the file or running it from the Command Prompt.

echo =======================================================
echo Starting dependency installation for Fake News Detector
echo =======================================================

:: Step 1: Install the required Python libraries using pip
echo.
echo [STEP 1/2] Installing pandas, scikit-learn, and nltk...
:: The 'py' command is the modern Python launcher on Windows.
py -m pip install pandas scikit-learn nltk

if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python libraries. Please check your Python/pip installation.
    pause
    exit /b %errorlevel%
)
echo Libraries installed successfully.
echo.

:: Step 2: Download all required NLTK data packages in one robust command
echo [STEP 2/2] Downloading NLTK data (punkt, wordnet, stopwords)...
py -c "import nltk; nltk.download(['punkt', 'wordnet', 'stopwords'])"

if %errorlevel% neq 0 (
    echo ERROR: Failed to download NLTK data. Please check your internet connection.
    pause
    exit /b %errorlevel%
)
echo.

echo =======================================================
echo Setup complete! All dependencies are now installed.
echo You can now run the main project file.
echo =======================================================
pause

