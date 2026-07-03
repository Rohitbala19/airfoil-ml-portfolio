#!/bin/bash

# Initialize local git repository
echo "Initializing local Git repository..."
git init

# Add all files to staging, respecting .gitignore
echo "Staging files..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Complete Airfoil ML Surrogate Model codebase"

# Rename default branch to main
git branch -M main

echo ""
echo "========================================================================"
echo "🎉 Local Git Repository successfully initialized and committed!"
echo "========================================================================"
echo ""
echo "To publish this project to your GitHub account:"
echo "1. Go to https://github.com/new and create a new public or private repository."
echo "   Name it: 'airfoil-ml-portfolio'"
echo "   IMPORTANT: Leave the options 'Add a README', 'Add .gitignore', and 'Choose a license' UNCHECKED."
echo ""
echo "2. Link your local repository to GitHub and push your code:"
echo "   Run the following commands in your terminal:"
echo "   "
echo "   git remote add origin https://github.com/<your-github-username>/airfoil-ml-portfolio.git"
echo "   git push -u origin main"
echo ""
echo "========================================================================"
