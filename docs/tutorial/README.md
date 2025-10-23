# RM-Gallery Tutorials

Welcome to the RM-Gallery tutorial series! This directory contains comprehensive guides to help you master reward models.

## 🗺️ Learning Paths

### 🌱 Beginner Path

**Goal**: Get started with reward models in 30 minutes

1. **[Quickstart Guide](../quickstart.md)** (5 min)
   - Install RM-Gallery
   - Use your first reward model
   - Evaluate AI responses

2. **[Building RM Overview](building_rm/overview.md)** (10 min)
   - Understand reward model types
   - Learn the architecture
   - See examples

3. **[Using Built-in RMs](building_rm/ready2use_rewards.md)** (15 min)
   - Explore 35+ pre-built models
   - Choose the right model
   - Run evaluations

### 🚀 Intermediate Path

**Goal**: Build and customize reward models

1. **[Building Custom RMs](building_rm/custom_reward.md)** (30 min)
   - Create rule-based rewards
   - Build LLM-based rewards
   - Use the Rubric-Critic-Score paradigm

2. **[Data Pipeline](data/pipeline.md)** (20 min)
   - Load data from various sources
   - Process and transform data
   - Export to different formats

3. **[End-to-End Tutorial](end-to-end.md)** (30 min)
   - Build a complete reward model from scratch
   - Test and validate
   - Deploy and use

### 🎓 Advanced Path

**Goal**: Train, evaluate, and deploy at scale

1. **[Training RM Overview](training_rm/overview.md)** (15 min)
   - Understand training paradigms
   - Set up training environment
   - Choose training strategy

2. **[Training with VERL](training_rm/training_rm.md)** (60 min)
   - Prepare training data
   - Configure training
   - Launch distributed training

3. **[High-Performance Serving](rm_serving/rm_server.md)** (45 min)
   - Deploy RM as a service
   - Set up load balancing
   - Monitor performance

## 📚 Tutorial Catalog

### Building Reward Models

| Tutorial | Level | Time | Description |
|----------|-------|------|-------------|
| [Overview](building_rm/overview.md) | Beginner | 10 min | Introduction to building RMs |
| [Ready-to-Use RMs](building_rm/ready2use_rewards.md) | Beginner | 15 min | Using pre-built models |
| [Custom Rewards](building_rm/custom_reward.md) | Intermediate | 30 min | Building custom RMs |
| [Auto Rubric](building_rm/autorubric.md) | Advanced | 45 min | Automatic rubric generation |

### Training Reward Models

| Tutorial | Level | Time | Description |
|----------|-------|------|-------------|
| [Training Overview](training_rm/overview.md) | Intermediate | 15 min | Introduction to training |
| [Bradley-Terry RM](training_rm/bradley_terry_rm.md) | Advanced | 60 min | Training Bradley-Terry models |
| [SFT RM](training_rm/sft_rm.md) | Advanced | 45 min | Training with SFT |
| [RL Training](training_rm/training_rm.md) | Advanced | 90 min | Full RL-based training |

### Evaluating Reward Models

| Tutorial | Level | Time | Description |
|----------|-------|------|-------------|
| [Evaluation Overview](evaluation/overview.md) | Beginner | 10 min | Introduction to evaluation |
| [RewardBench2](evaluation/rewardbench2.md) | Intermediate | 30 min | Latest benchmark |
| [Conflict Detector](evaluation/conflict_detector.md) | Advanced | 45 min | Detect evaluation conflicts |
| [JudgeBench](evaluation/judgebench.md) | Intermediate | 30 min | Judge capability testing |
| [RM-Bench](evaluation/rmbench.md) | Intermediate | 30 min | Comprehensive evaluation |

### Data Processing

| Tutorial | Level | Time | Description |
|----------|-------|------|-------------|
| [Data Pipeline](data/pipeline.md) | Beginner | 20 min | Complete data workflow |
| [Data Annotation](data/annotation.md) | Intermediate | 30 min | Annotating training data |
| [Data Loading](data/load.md) | Beginner | 15 min | Loading from various sources |
| [Data Processing](data/process.md) | Intermediate | 25 min | Transforming data |

### Applications

| Tutorial | Level | Time | Description |
|----------|-------|------|-------------|
| [RM Server](rm_serving/rm_server.md) | Advanced | 45 min | Deploy RM as service |
| [Best-of-N](rm_application/best_of_n.md) | Intermediate | 20 min | Select best response |
| [Data Refinement](rm_application/data_refinement.md) | Intermediate | 30 min | Improve data quality |
| [Post Training](rm_application/post_training.md) | Advanced | 60 min | RLHF integration |

## 🎯 By Use Case

### I want to...

**Evaluate AI responses**
→ Start with [Quickstart](../quickstart.md)
→ Then [Using Built-in RMs](building_rm/ready2use_rewards.md)

**Build a custom reward model**
→ Read [Building Custom RMs](building_rm/custom_reward.md)
→ Try [End-to-End Tutorial](end-to-end.md)

**Train my own reward model**
→ Start with [Training Overview](training_rm/overview.md)
→ Then [RL Training](training_rm/training_rm.md)

**Test on benchmarks**
→ Read [Evaluation Overview](evaluation/overview.md)
→ Try specific benchmarks (RewardBench2, RM-Bench, etc.)

**Deploy to production**
→ Follow [RM Server Guide](rm_serving/rm_server.md)
→ Implement [Best-of-N](rm_application/best_of_n.md)

**Process custom data**
→ Read [Data Pipeline](data/pipeline.md)
→ Use [Data Loading](data/load.md)

## 💡 Tutorial Tips

### Before You Start

- ✅ Install RM-Gallery: `pip install rm-gallery`
- ✅ Set up Python environment (>= 3.10, < 3.13)
- ✅ (Optional) Get API credentials for LLM-based models

### While Learning

- 📖 **Read in order**: Tutorials build on each other
- 💻 **Run the code**: Try examples in your environment
- 🔄 **Experiment**: Modify code and see what happens
- ❓ **Ask questions**: Use GitHub Discussions

### After Completing

- 🎯 **Apply to your project**: Use what you learned
- 🤝 **Share feedback**: Help us improve tutorials
- 📝 **Contribute**: Add your own examples

## 🔗 Quick Links

### Essential

- [Quickstart Guide](../quickstart.md) - Get started in 5 minutes
- [FAQ](../faq.md) - Common questions answered
- [API Reference](../api_reference.md) - Complete API docs

### Interactive

- [Jupyter Notebooks](../../examples/) - Hands-on tutorials
- [End-to-End Tutorial](end-to-end.md) - Complete project

### Reference

- [RM Library](../library/rm_library.md) - All available models
- [Rubric Library](../library/rubric_library.md) - Evaluation rubrics
- [Contribution Guide](../contribution.md) - How to contribute

## 📊 Tutorial Difficulty Legend

- 🌱 **Beginner**: No prior experience needed
- 🚀 **Intermediate**: Basic understanding required
- 🎓 **Advanced**: In-depth knowledge helpful

## ⏱️ Time Estimates

Time estimates are for:
- **Reading**: Understanding the concepts
- **Coding**: Running and modifying examples
- **Practice**: Experimenting with your own data

Actual time may vary based on your experience level.

## 🆘 Getting Help

**Stuck on a tutorial?**

1. Check the [FAQ](../faq.md) first
2. Search [GitHub Issues](https://github.com/modelscope/RM-Gallery/issues)
3. Ask in [GitHub Discussions](https://github.com/modelscope/RM-Gallery/discussions)
4. Join our community channels

**Found an error?**

Please report it by:
1. Opening a GitHub Issue
2. Including the tutorial name
3. Describing the problem
4. Suggesting a fix (optional)

## 🎓 Additional Resources

### External Learning

- **OpenAI Evals**: Similar evaluation framework
- **RLHF Papers**: Academic background
- **LLM Alignment**: Broader context

### Community

- **GitHub**: Source code and issues
- **Discussions**: Q&A and ideas
- **Examples**: Community contributions

## 🚀 Next Steps

After completing the tutorials:

1. **Build your first project** using RM-Gallery
2. **Share your experience** with the community
3. **Contribute back** with examples or improvements
4. **Stay updated** on new features and models

---

**Ready to start?** Go to the [Quickstart Guide](../quickstart.md) 🎉

**Have questions?** Check the [FAQ](../faq.md) or ask in [Discussions](https://github.com/modelscope/RM-Gallery/discussions) 💬

**Want to contribute?** Read our [Contribution Guide](../contribution.md) 🤝

