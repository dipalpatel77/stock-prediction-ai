# .gitignore Guide for AI Stock Predictor

This document explains the comprehensive `.gitignore` file and why certain files are excluded from version control.

## 📁 File Categories Excluded

### 🔒 **Security & Credentials**
```
# API Keys and Secrets
.env
.env.local
.env.production.local
secrets.json
credentials.json
api_keys.json
*.key
*.pem
*.crt
```
**Why**: Never commit API keys, passwords, or sensitive configuration data.

### 📊 **Data Files**
```
# Stock and Financial Data
stock_data/
financial_data/
market_data/
historical_data/
angel_data/
*.csv
*.xlsx
*.parquet
*.feather
```
**Why**: Data files are large, change frequently, and should be downloaded fresh.

### 🤖 **ML Models & Artifacts**
```
# Model Files
*.pkl
*.pickle
*.joblib
*.h5
*.hdf5
models/
trained_models/
```
**Why**: Model files are large, generated from code, and can be recreated.

### 💾 **Cache & Temporary Files**
```
# Cache Directories
cache/
.cache/
__pycache__/
*.tmp
*.temp
*.log
```
**Why**: Cache files are temporary and can be regenerated.

### 🗄️ **Database Files**
```
# Database Files
*.db
*.sqlite
*.sqlite3
database/
db/
```
**Why**: Database files contain runtime data and should be managed separately.

## 🎯 **Project-Specific Exclusions**

### **AI/ML Specific**
- **Model files**: `.pkl`, `.h5`, `.joblib` - Generated models
- **Training data**: `training_data/`, `validation_data/` - Large datasets
- **Checkpoints**: `checkpoints/`, `saved_models/` - Model checkpoints

### **Financial Data**
- **Stock data**: `stock_data/`, `angel_data/` - Downloaded stock data
- **Market data**: `market_data/`, `historical_data/` - Market information
- **Data formats**: `.csv`, `.xlsx`, `.parquet` - Data file formats

### **Analysis Outputs**
- **Reports**: `reports/`, `analysis/` - Generated reports
- **Predictions**: `predictions/`, `results/` - Prediction outputs
- **Logs**: `logs/`, `*.log` - System logs

## 🔧 **Development Exclusions**

### **IDE Files**
```
.vscode/
.idea/
*.swp
*.swo
```
**Why**: IDE-specific settings that vary between developers.

### **Python Specific**
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
dist/
*.egg-info/
```
**Why**: Python bytecode and build artifacts.

### **Virtual Environments**
```
venv/
env/
.venv/
.env/
```
**Why**: Virtual environments are local and should be recreated.

## 📈 **Performance Considerations**

### **Large Files**
```
*.zip
*.tar.gz
*.rar
*.7z
```
**Why**: Archive files are large and can be recreated.

### **Media Files**
```
*.png
*.jpg
*.jpeg
*.gif
*.svg
*.mp4
*.avi
```
**Why**: Media files are large and not essential for code functionality.

## 🛡️ **Security Best Practices**

### **Never Commit**
- API keys and secrets
- Database credentials
- SSL certificates
- Personal configuration

### **Environment Files**
```
.env
.env.local
.env.production.local
```
**Why**: Contains sensitive environment-specific configuration.

## 🔍 **What TO Include**

### **Configuration Templates**
```
config.example.json
.env.example
requirements.txt
```

### **Documentation**
```
README.md
docs/
*.md
```

### **Source Code**
```
*.py
*.js
*.html
*.css
```

### **Configuration Files**
```
setup.py
pyproject.toml
Dockerfile
docker-compose.yml
```

## 🚀 **Quick Reference**

### **Common Commands**
```bash
# Check what files are being tracked
git status

# Check what files are ignored
git check-ignore -v <file>

# Add file to .gitignore
echo "filename" >> .gitignore

# Remove file from tracking (but keep local)
git rm --cached <file>
```

### **Troubleshooting**
```bash
# If you accidentally committed sensitive data
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch secrets.json' \
--prune-empty --tag-name-filter cat -- --all

# Remove file from all history
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch <file>' \
--prune-empty --tag-name-filter cat -- --all
```

## 📋 **Checklist for New Files**

Before adding files to the repository, ask:

1. **Is it sensitive?** → Add to .gitignore
2. **Is it generated?** → Add to .gitignore
3. **Is it large?** → Consider .gitignore
4. **Is it temporary?** → Add to .gitignore
5. **Is it essential for the project?** → Include in repo

## 🔄 **Maintenance**

### **Regular Tasks**
- Review .gitignore quarterly
- Remove outdated patterns
- Add new file types as needed
- Check for accidentally committed sensitive data

### **Team Guidelines**
- Document new .gitignore rules
- Use consistent patterns
- Test .gitignore changes
- Keep .gitignore organized

## ⚠️ **Important Notes**

1. **Never commit sensitive data** - Even if removed later, it's in git history
2. **Test .gitignore changes** - Use `git check-ignore` to verify
3. **Document exceptions** - If you need to include normally ignored files
4. **Keep it organized** - Use comments and sections for clarity

## 🆘 **Emergency Procedures**

### **If Sensitive Data is Committed**
1. **Immediately**: Change passwords/API keys
2. **Remove from history**: Use `git filter-branch`
3. **Force push**: `git push --force-with-lease`
4. **Notify team**: Alert about compromised credentials

### **If Large Files are Committed**
1. **Remove from tracking**: `git rm --cached <file>`
2. **Add to .gitignore**: `echo "*.large" >> .gitignore`
3. **Commit changes**: `git commit -m "Remove large files"`
4. **Use Git LFS**: For legitimate large files

---

**Remember**: A good .gitignore is essential for a clean, secure, and efficient repository! 🎯
