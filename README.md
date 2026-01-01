
# 🎥 LLM YouTube Content Generator – LangChain Workflow

## 📌 Overview

This project implements a **complete end-to-end AI workflow** to automate metadata generation for YouTube videos using **Large Language Models (LLMs)** and **LangChain**. Built during my internship at **Oladoc**, the system processes a YouTube video, extracts and refines the transcript, and generates optimized content including:

- 🎯 Video Title
- 📝 Description (using structured template)
- 🔍 SEO Keywords

---

## 🎯 Objective

To streamline YouTube content optimization by leveraging LLMs for transcription refinement, context understanding, and intelligent metadata generation — enabling faster publishing and better discoverability.

---

## 🧠 Workflow Architecture

```mermaid
graph TD;
    A[YouTube Video] --> B[Audio Extraction & Chunking];
    B --> C[Speech-to-Text Transcription (STT)];
    C --> D[Gemini LLM – Transcript Refinement];
    D --> E[Gemini LLM – Title, Description, Keywords];
    E --> F[Final Output (Text + JSON)];
