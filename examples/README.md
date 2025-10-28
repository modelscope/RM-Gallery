# RM-Gallery Examples

Welcome to the RM-Gallery examples! This directory contains interactive Jupyter notebooks and practical examples to help you get started with reward models.

## 📚 Interactive Notebooks

### 🚀 [quickstart.ipynb](quickstart.ipynb)
**Your first reward model in 5 minutes**

Perfect for beginners! Learn how to:
- Install and set up RM-Gallery
- Use pre-built reward models
- Evaluate AI responses
- Interpret reward scores

**Time to complete:** 5 minutes

---

### 🛠️ [custom-rm.ipynb](custom-rm.ipynb)
**Build your own reward models**

Learn advanced techniques:
- Understanding the reward model architecture
- Creating rule-based reward models
- Building LLM-based reward models
- Using the Rubric-Critic-Score paradigm

**Prerequisites:** Complete quickstart.ipynb first
**Time to complete:** 15 minutes

---

### 🧪 [evaluation.ipynb](evaluation.ipynb)
**Complete evaluation pipeline**

Run comprehensive evaluations:
- Loading benchmark datasets
- Batch evaluation with parallel processing
- Calculating metrics
- Analyzing results

**Prerequisites:** Understand basic RM concepts
**Time to complete:** 10 minutes

---

## 📁 Code Examples

### Data Processing
- **[data/data_pipeline.py](data/data_pipeline.py)** - Complete data pipeline example
- **[data/data_from_yaml.py](data/data_from_yaml.py)** - Load data from YAML config

### Training
- **[train/pointwise/](train/pointwise/)** - Train pointwise reward models
- **[train/pairwise/](train/pairwise/)** - Train pairwise reward models
- **[train/bradley-terry/](train/bradley-terry/)** - Bradley-Terry model training

### Applications
- **[rm_application/data_correction.py](rm_application/data_correction.py)** - Data correction with RM

### Rubric Generation
- **[rubric/auto_rubric.py](rubric/auto_rubric.py)** - Automatic rubric generation
- **[rubric/run_rubric_generator.py](rubric/run_rubric_generator.py)** - Rubric generator script

---

## 🎯 Quick Start

### Option 1: Run Notebooks Locally

```bash
# Install RM-Gallery
pip install rm-gallery

# Install Jupyter
pip install jupyter

# Launch Jupyter
jupyter notebook

# Open any .ipynb file
```

### Option 2: Run in Colab

Click the "Open in Colab" badge in each notebook (coming soon).

### Option 3: Run in VS Code

1. Install the Jupyter extension for VS Code
2. Open any `.ipynb` file
3. Click "Select Kernel" and choose your Python environment

---

## 🗺️ Learning Path

**Beginner Path:**
1. Start with [quickstart.ipynb](quickstart.ipynb)
2. Read the [Quickstart Guide](https://modelscope.github.io/RM-Gallery/quickstart/)
3. Explore pre-built reward models

**Intermediate Path:**
1. Try [custom-rm.ipynb](custom-rm.ipynb)
2. Build your first custom reward model
3. Read the [Custom RM Guide](https://modelscope.github.io/RM-Gallery/tutorial/building_rm/custom_reward/)

**Advanced Path:**
1. Complete [evaluation.ipynb](evaluation.ipynb)
2. Train your own reward model (see [train/](train/))
3. Deploy to production (see [RM Serving Guide](https://modelscope.github.io/RM-Gallery/tutorial/rm_serving/rm_server/))

---

## 💡 Tips

- **Start Simple**: Begin with the quickstart notebook
- **Run Cells**: Execute cells one by one to understand each step
- **Experiment**: Modify the code and see what happens
- **Check Docs**: Refer to the [documentation](https://modelscope.github.io/RM-Gallery/) when needed

---

## 🐛 Troubleshooting

### Common Issues

**Problem: Import errors**
```bash
# Solution: Make sure RM-Gallery is installed
pip install rm-gallery
```

**Problem: API errors for LLM-based models**
```python
# Solution: Set up your API credentials
import os
os.environ["OPENAI_API_KEY"] = "your_api_key"
os.environ["BASE_URL"] = "your_base_url"
```

**Problem: Notebook kernel crashes**
```bash
# Solution: Restart the kernel
# In Jupyter: Kernel → Restart
# In VS Code: Click the restart button
```

---

## 🤝 Contributing

Have a great example? We'd love to include it!

1. Fork the repository
2. Create your example
3. Add documentation
4. Submit a pull request

See our [Contribution Guide](../docs/contribution.md) for details.

---

## 📞 Need Help?

- 📚 [Full Documentation](https://modelscope.github.io/RM-Gallery/)
- 💬 [GitHub Discussions](https://github.com/modelscope/RM-Gallery/discussions)
- 🐛 [Report Issues](https://github.com/modelscope/RM-Gallery/issues)

---

Happy learning! 🚀

