# Abhilekh

Abhilekh is a document intelligence pipeline for extracting structured,
searchable information from Indian Government Gazette notifications.

Government Gazette documents often contain complex layouts, Hindi and
English text, tables, scanned pages, and imperfect PDF text layers.
Abhilekh explores how OCR, document processing, and agentic extraction
can convert these documents into reliable structured records.

## Problem

Government Gazette notifications contain valuable information such as:

- Ministries and departments
- Notification numbers
- Dates
- Subjects
- Regulatory changes
- Appointments
- Amendments
- Official forms and notices

However, extracting this information automatically is difficult because
Gazette PDFs can contain:

- Scanned pages
- Hindi and English text
- Complex layouts
- Tables
- Corrupted PDF text layers
- OCR errors
- Inconsistent document structures

Abhilekh aims to build a robust pipeline for turning these documents
into structured and searchable government records.

## Architecture

The planned pipeline is:

    Gazette PDF
         |
         v
    PDF Inspection
         |
         v
    Native Text Extraction
         |
         +----------------+
         |                |
         v                v
    Usable Text?       OCR Pipeline
         |                |
         +-------+--------+
                 |
                 v
        Document Extraction
                 |
                 v
        Agentic Validation
                 |
                 v
        Structured Records
                 |
                 v
            PostgreSQL
                 |
                 v
             FastAPI
                 |
                 v
        Searchable Gazette API

## Current Progress

### Completed

- Collected real Indian Government Gazette PDFs
- Built a PDF inspection utility using PyMuPDF
- Implemented native PDF text extraction
- Installed Tesseract OCR
- Added Hindi and English OCR support
- Built an initial OCR pipeline
- Compared native PDF extraction with OCR output on Gazette documents

### In Progress

- OCR vs. native extraction evaluation
- Extraction quality metrics
- Gazette field identification
- Structured document extraction

### Planned

- Agentic schema-based extraction
- OCR/native text verification
- Self-correction and validation
- PostgreSQL storage
- FastAPI search and query service
- Dockerization
- Deployment

## Example

A Gazette notification can contain information such as:

    Ministry:
    Ministry of Commerce and Industry

    Department:
    Department of Commerce

    Notification Number:
    29/2026-27

    Date:
    18 August 2026

    Subject:
    Amendment in Import Policy and Policy condition
    of Clear Float Glass

Abhilekh aims to automatically identify and structure fields such as
these from raw Gazette documents.

## OCR Challenge

A key challenge observed during development is that native PDF text
extraction and OCR can produce different errors.

For example, native extraction may produce corrupted Hindi text while
OCR can recover the intended characters. However, OCR can also introduce
its own recognition errors.

Therefore, Abhilekh will evaluate both sources rather than assuming
that OCR is always better.

## Evaluation

The planned evaluation will include:

- OCR character error rate
- Field-level precision
- Field-level recall
- Extraction accuracy
- Comparison against a manually labeled benchmark
- Comparison between a non-agentic baseline and the proposed
  agentic extraction pipeline

## Tech Stack

### Current

- Python
- PyMuPDF
- Tesseract OCR
- pytesseract
- Pillow
- pandas

### Planned

- LangChain
- PostgreSQL
- FastAPI
- Docker
- Cloud deployment

## Project Structure

    Abhilekh/
    |
    ├── data/
    │   └── raw/
    │       ├── gazettes/
    │       ├── text/
    │       └── ocr/
    |
    ├── src/
    │   ├── ingestion/
    │   │   ├── inspect_pdf.py
    │   │   └── extract_text.py
    │   │
    │   └── ocr/
    │       └── test_ocr.py
    |
    ├── tests/
    ├── .gitignore
    └── README.md

## Data

The project uses publicly available Indian Government Gazette
documents for experimentation.

Raw Gazette PDFs and generated extraction outputs are kept locally
and are not committed to the repository.

## Goal

The long-term goal of Abhilekh is to provide a reliable pipeline that
can transform messy government documents into structured, validated,
and searchable records.
