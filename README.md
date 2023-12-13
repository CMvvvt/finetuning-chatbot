# Finetuning pre-trained LLM model to build chatbot

# Usage Guide

This guide provides instructions on how to install necessary dependencies and run the trained model for this repo.

## Installation

```bash
pip install transformers datasets accelerate peft trl
```

## Running

A chatbot interface should show up after this command.

```bash
python running.py
```

# Dataset Description: openassistant-guanaco

## Overview

- **Source**: Hugging Face
- **Accessibility**: Public
- **Dataset URL**: [openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

## Background

- **Creator**: [timdettmers](https://huggingface.co/timdettmers)
- **Description**: This dataset is a subset of the [Open Assistant dataset](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/main)
  This subset of the data only contains the highest-rated paths in the conversation tree

- **Primary Use Case**: [Primary use case of the dataset]

## Basic Statistics

- **Total Data Points**: 10,364
- **Dataset Size**: 22 MB

# Implementation:

This is the implementation description, for instruction to use this trained model, please check [Usage](#).

## Environment Setup

- **Device Configuration**: Utilizes CUDA if available, otherwise CPU.
- **Dependencies**: `torch`, `transformers`, `trl`, `datasets`.

## Dataset Preparation

- **Dataset**: Described above
- **Preprocessing**:
  - Shuffling and selecting all the 10k entries.

## Model Configuration

- **Base Model**: EleutherAI's GPT-Neo 1.3B.
- **Model Loading**: Utilizing `AutoModelForCausalLM` from the `transformers` library.
  - **Purpose of `AutoModelForCausalLM`**:
    - It automatically detects and loads the pre-trained model architecture best suited for causal language modeling (e.g., GPT-Neo).
    - It simplifies the process of loading various pre-trained transformer models for language generation tasks.
- **Tokenizer**: Fast tokenizer enabled.
- **Model Adjustments**:
  - Addition of LoRA (Low-Rank Adaptation) parameters.
  - Adjustment of pad tokens.
  - Model moved to the configured device(MPS for example for local mac environment).

## Training Configuration

- **Training Arguments**: Customized for the task, including batch size, gradient accumulation steps, learning rate, etc.
- **Batch Size**: 4 per device.
- **Gradient Accumulation Steps**: 4.
- **Optimizer**: AdamW.
- **Learning Rate**: 2e-4.
- **Max Steps**: 500.
- **Warmup Ratio**: 0.03.
- **Scheduler Type**: Constant.

## Training Process

- **Trainer Initialization**: Using `SFTTrainer` from `trl`.
- **Maximum Sequence Length**: 512.
- **Training Execution**: Model trained on the preprocessed dataset.

## Post-Training

- **Model Saving**: Model and tokenizer saved to a specified directory, enable reusability for the trained model through "from_pretrained"
