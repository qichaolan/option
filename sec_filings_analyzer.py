#!/usr/bin/env python3
"""
SEC Filings Analyzer with AI-Powered Insights

Analyzes SEC filing PDFs (10-K, 10-Q) using OpenAI to generate comprehensive analysis reports.

Features:
- Extracts Table of Contents from PDF
- Generates intelligent analysis plan based on filing structure
- Processes content in priority-based batches
- Uses AI to analyze each section
- Generates final synthesis report
- Exports professional PDF reports

Usage:
    python sec_filings_analyzer.py --pdf /path/to/10k.pdf

Requirements:
    pip install pdfplumber pypdf openai pyyaml reportlab
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# ============================================================================
# CONSTANTS
# ============================================================================

# Common SEC 10-K/10-Q section patterns for TOC inference
SEC_SECTION_PATTERNS = [
    r'^\s*PART\s+[IVX]+\b',
    r'^\s*Item\s+\d+[A-Z]?\.',
    r'^\s*Item\s+\d+[A-Z]?\b',
    r"Management['\u2019]?s\s+Discussion\s+and\s+Analysis",
    r'Risk\s+Factors',
    r'Financial\s+Statements',
    r'Consolidated\s+Statements?\s+of\s+(Operations|Income|Cash\s+Flows?|Financial\s+Position)',
    r'Consolidated\s+Balance\s+Sheets?',
    r'Notes?\s+to\s+(Consolidated\s+)?Financial\s+Statements',
    r'Liquidity\s+and\s+Capital\s+Resources',
    r'Critical\s+Accounting\s+(Policies|Estimates)',
    r'Segment\s+Information',
    r'Business\s+Overview',
    r'Results\s+of\s+Operations',
]


# ============================================================================
# TOC EXTRACTOR
# ============================================================================

class TOCExtractor:
    """Extracts Table of Contents and metadata from SEC filing PDFs."""

    def __init__(self, pdf_path: Path):
        """
        Initialize TOC extractor.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = pdf_path

    def extract_structured_data(self) -> dict:
        """
        Extract structured data including filing metadata and TOC.

        Returns:
            Dict with filing metadata and table of contents
        """
        # Extract metadata
        metadata = self._extract_metadata()

        # Extract TOC
        toc = self._extract_explicit_toc()

        # Fallback to inferring TOC if explicit extraction fails
        if not toc:
            toc = self._infer_toc(silent=True)

        if not toc:
            raise ValueError("Could not extract or infer TOC from PDF")



        # Build structured data
        structured_data = {
            "filing": {
                "document_title": metadata.get("document_title", ""),
                "filename": self.pdf_path.name,
                "title": metadata.get("title", ""),
                "subject": metadata.get("subject", ""),
                "author": metadata.get("author", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creation_date", ""),
                "total_pages": metadata.get("total_pages", 0)
            },
            "table_of_contents": toc
        }

        return structured_data

    def _extract_metadata(self) -> dict:
        """Extract PDF metadata including document title from TOC page."""
        metadata = {}

        if HAS_PDFPLUMBER:
            with pdfplumber.open(self.pdf_path) as pdf:
                metadata["total_pages"] = len(pdf.pages)

                # Extract PDF metadata
                if pdf.metadata:
                    metadata["title"] = pdf.metadata.get("Title", "")
                    metadata["subject"] = pdf.metadata.get("Subject", "")
                    metadata["author"] = pdf.metadata.get("Author", "")
                    metadata["creator"] = pdf.metadata.get("Creator", "")
                    metadata["producer"] = pdf.metadata.get("Producer", "")

                    # Handle creation date
                    creation_date = pdf.metadata.get("CreationDate", "")
                    if creation_date:
                        metadata["creation_date"] = str(creation_date)
                    else:
                        metadata["creation_date"] = ""

                # Extract document title from TOC page (usually page 2)
                # Look for company name and filing type header
                metadata["document_title"] = self._extract_document_title(pdf)

        elif HAS_PYPDF:
            reader = PdfReader(str(self.pdf_path))
            metadata["total_pages"] = len(reader.pages)
            if reader.metadata:
                metadata["title"] = reader.metadata.get("/Title", "")
                metadata["subject"] = reader.metadata.get("/Subject", "")
                metadata["author"] = reader.metadata.get("/Author", "")
                metadata["creator"] = reader.metadata.get("/Creator", "")
                metadata["producer"] = reader.metadata.get("/Producer", "")

                # Handle creation date
                creation_date = reader.metadata.get("/CreationDate", "")
                if creation_date:
                    metadata["creation_date"] = str(creation_date)
                else:
                    metadata["creation_date"] = ""

            # Extract document title
            metadata["document_title"] = self._extract_document_title_pypdf(reader)

        return metadata

    def _extract_document_title(self, pdf) -> str:
        """Extract document title (company name + filing type) from PDF pages."""
        # Check first few pages for company name and filing type
        for page_num in range(min(5, len(pdf.pages))):
            page = pdf.pages[page_num]
            text = page.extract_text()

            if not text:
                continue

            lines = text.split('\n')

            # Look for patterns like:
            # "TESLA, INC."
            # "FORM 10-Q FOR THE QUARTER ENDED SEPTEMBER 30, 2025"
            # These usually appear together at the top of the TOC page

            for i, line in enumerate(lines):
                line_stripped = line.strip()

                # Check if this looks like a company name (all caps, ends with INC., LLC., etc.)
                if re.match(r'^[A-Z][A-Z\s,\.&]+(?:INC\.|LLC\.|CORP\.|LTD\.|L\.P\.)$', line_stripped):
                    company_name = line_stripped

                    # Look for filing type in next few lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if re.match(r'^FORM\s+\d+[A-Z-]*', next_line, re.IGNORECASE):
                            filing_info = next_line
                            return f"{company_name}, {filing_info}"

        return ""

    def _extract_document_title_pypdf(self, reader) -> str:
        """Extract document title using pypdf."""
        # Similar logic for pypdf
        for page_num in range(min(5, len(reader.pages))):
            page = reader.pages[page_num]
            text = page.extract_text()

            if not text:
                continue

            lines = text.split('\n')

            for i, line in enumerate(lines):
                line_stripped = line.strip()

                if re.match(r'^[A-Z][A-Z\s,\.&]+(?:INC\.|LLC\.|CORP\.|LTD\.|L\.P\.)$', line_stripped):
                    company_name = line_stripped

                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if re.match(r'^FORM\s+\d+[A-Z-]*', next_line, re.IGNORECASE):
                            filing_info = next_line
                            return f"{company_name}, {filing_info}"

        return ""

    def _extract_explicit_toc(self) -> Optional[List[dict]]:
        """Extract explicit TOC if present."""
        if HAS_PDFPLUMBER:
            return self._extract_toc_pdfplumber()
        elif HAS_PYPDF:
            return self._extract_toc_pypdf()
        else:
            raise ImportError("No PDF library available (need pdfplumber or pypdf)")

    def _extract_toc_pdfplumber(self) -> Optional[List[dict]]:
        """Extract TOC using pdfplumber."""
        toc_entries = []

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            found_toc_page = False

            # Look for TOC in first 20 pages
            for page_num in range(min(20, len(pdf.pages))):
                # Skip if we already found and parsed a TOC
                if found_toc_page and toc_entries:
                    break

                page = pdf.pages[page_num]
                text = page.extract_text()

                if not text:
                    continue

                # Look for TOC indicators (INDEX or TABLE OF CONTENTS)
                # But make sure it's an actual TOC page, not just a page with "Table of Contents" in header
                # An actual TOC should have multiple Item entries
                if re.search(r'(index|table\s+of\s+contents)', text, re.IGNORECASE):
                    # Check if this looks like an actual TOC page (has multiple Items)
                    item_count = len(re.findall(r'Item\s+\d+[A-Z]?\.', text, re.IGNORECASE))
                    if item_count < 3:
                        # Not a real TOC page, skip it
                        continue

                    found_toc_page = True
                    # Try to parse TOC entries
                    lines = text.split('\n')
                    for line in lines:
                        line_stripped = line.strip()

                        # Skip empty lines and common headers
                        if not line_stripped or line_stripped.lower() in ['index', 'table of contents', 'page']:
                            continue

                        # Match SEC filing TOC patterns:
                        # 1. "PART I. FINANCIAL INFORMATION" (no page number)
                        # 2. "PART II. OTHER INFORMATION" (no page number)
                        # 3. "Item 1. Financial Statements 4"
                        # 4. "Item 1A. Risk Factors 39"
                        # 5. "Consolidated Balance Sheets 4" (sub-item under Item 1)
                        # 6. "Notes to Consolidated Financial Statements 10"
                        # 7. "Signatures 41"

                        # First, capture PART headers without page numbers
                        part_match = re.match(r'^(PART\s+[IVX]+\.\s*.+)$', line_stripped, re.IGNORECASE)
                        if part_match and not re.search(r'\d+$', line_stripped):
                            # This is a PART header with no page number
                            toc_entries.append({
                                'title': part_match.group(1).strip(),
                                'page_start': None,
                                'page_end': None,
                                'level': 0  # Top level
                            })
                            continue

                        # Capture Item entries (with or without dots)
                        # "Item 1. Financial Statements 4" or "Item 1A. Risk Factors 39"
                        item_match = re.match(r'^(Item\s+\d+[A-Z]?\..+?)\s+(\d+)$', line_stripped, re.IGNORECASE)
                        if item_match:
                            title = item_match.group(1).strip()
                            page = int(item_match.group(2))
                            toc_entries.append({
                                'title': title,
                                'page_start': page,
                                'page_end': page,
                                'level': 1  # Main item level
                            })
                            # Track current item for sub-items
                            current_item = re.match(r'^Item\s+(\d+[A-Z]?)\.', title, re.IGNORECASE).group(1)
                            continue

                        # Capture Signatures entry (common at end of SEC filings) - before sub-items
                        # This prevents "Signatures" from being treated as a sub-item
                        if re.match(r'^Signatures\s+\d+$', line_stripped, re.IGNORECASE):
                            sig_match = re.match(r'^(Signatures)\s+(\d+)$', line_stripped, re.IGNORECASE)
                            title = sig_match.group(1).strip()
                            page = int(sig_match.group(2))
                            toc_entries.append({
                                'title': title,
                                'page_start': page,
                                'page_end': page,
                                'level': 1
                            })
                            # Reset current_item since we're past all Items
                            if 'current_item' in locals():
                                del current_item
                            continue

                        # Capture sub-items (indented entries under Items)
                        # Look for lines with text and page number that aren't Items or Parts
                        # Common patterns: "Consolidated Balance Sheets 4", "Notes to Financial Statements 10"
                        sub_match = re.match(r'^([A-Z][^0-9]+?)\s+(\d+)$', line_stripped)
                        if sub_match:
                            title = sub_match.group(1).strip()
                            page = int(sub_match.group(2))
                            # Skip if it's a header or common non-content line
                            if title.lower() not in ['page', 'index', 'table of contents', 'signatures']:
                                # This is a sub-item, add current item prefix
                                if 'current_item' in locals():
                                    full_title = f"Item {current_item}. {title}"
                                    toc_entries.append({
                                        'title': full_title,
                                        'page_start': page,
                                        'page_end': page,
                                        'level': 2  # Sub-item level
                                    })
                            continue

            # Fill in page_end values (only for entries with actual page numbers)
            if toc_entries:
                # Get indices of entries with page numbers
                entries_with_pages = [(i, entry) for i, entry in enumerate(toc_entries) if entry['page_start'] is not None]

                # Update page_end for entries with pages
                # For main items (level 1), calculate based on next main item
                # For sub-items (level 2), calculate based on next sub-item or parent's end
                for idx, (i, entry) in enumerate(entries_with_pages):
                    page_start = entry['page_start']
                    level = entry.get('level', 1)

                    if level == 1:
                        # Main item: find next main item (level 0 or 1)
                        next_main_idx = None
                        for future_idx in range(idx + 1, len(entries_with_pages)):
                            future_i, future_entry = entries_with_pages[future_idx]
                            if future_entry.get('level', 1) <= 1:
                                next_main_idx = future_idx
                                break

                        if next_main_idx is not None:
                            next_page = entries_with_pages[next_main_idx][1]['page_start']
                            toc_entries[i]['page_end'] = max(page_start, next_page - 1)
                        else:
                            # Last main item goes to end of document
                            toc_entries[i]['page_end'] = total_pages
                    else:
                        # Sub-item (level 2): find next entry at any level
                        if idx < len(entries_with_pages) - 1:
                            next_page = entries_with_pages[idx + 1][1]['page_start']
                            toc_entries[i]['page_end'] = max(page_start, next_page - 1)
                        else:
                            # Last sub-item
                            toc_entries[i]['page_end'] = total_pages

        return toc_entries if toc_entries else None

    def _extract_toc_pypdf(self) -> Optional[List[dict]]:
        """Extract TOC using pypdf."""
        reader = PdfReader(str(self.pdf_path))

        # Try to use PDF outline/bookmarks
        if reader.outline:
            toc_entries = []
            for item in reader.outline:
                if isinstance(item, dict):
                    title = item.get('/Title', '')
                    page_obj = item.get('/Page')
                    if title and page_obj:
                        # Get page number
                        page_num = reader.pages.index(page_obj) + 1
                        toc_entries.append({
                            'title': title,
                            'page_start': page_num,
                            'page_end': page_num
                        })

            if toc_entries:
                # Fill in page_end
                for i in range(len(toc_entries) - 1):
                    toc_entries[i]['page_end'] = toc_entries[i + 1]['page_start'] - 1
                return toc_entries

        return None

    def _infer_toc(self, silent: bool = False) -> List[dict]:
        """
        Infer TOC by scanning for section headings.

        Args:
            silent: If True, suppress progress messages

        Looks for patterns like:
        - PART I, PART II
        - Item 1., Item 1A., Item 7.
        - Common section names
        """
        toc_entries = []

        if HAS_PDFPLUMBER:
            with pdfplumber.open(self.pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if not silent:
                    print(f"   Scanning {total_pages} pages for section headings...")

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Progress indicator
                    if not silent and page_num % 10 == 0:
                        print(f"   ... page {page_num}/{total_pages}")

                    text = page.extract_text()
                    if not text:
                        continue

                    lines = text.split('\n')
                    for line in lines:
                        line_stripped = line.strip()

                        # Check against patterns
                        for pattern in SEC_SECTION_PATTERNS:
                            if re.search(pattern, line_stripped, re.IGNORECASE):
                                # Found a section heading
                                title = line_stripped

                                # Update previous entry's page_end
                                if toc_entries:
                                    toc_entries[-1]['page_end'] = page_num - 1

                                toc_entries.append({
                                    'title': title,
                                    'page_start': page_num,
                                    'page_end': page_num
                                })
                                break

                # Set final page_end
                if toc_entries:
                    toc_entries[-1]['page_end'] = total_pages

        elif HAS_PYPDF:
            reader = PdfReader(str(self.pdf_path))
            total_pages = len(reader.pages)
            if not silent:
                print(f"   Scanning {total_pages} pages for section headings...")

            for page_num in range(total_pages):
                # Progress indicator
                if not silent and (page_num + 1) % 10 == 0:
                    print(f"   ... page {page_num + 1}/{total_pages}")

                page = reader.pages[page_num]
                text = page.extract_text()

                if not text:
                    continue

                lines = text.split('\n')
                for line in lines:
                    line_stripped = line.strip()

                    for pattern in SEC_SECTION_PATTERNS:
                        if re.search(pattern, line_stripped, re.IGNORECASE):
                            title = line_stripped

                            if toc_entries:
                                toc_entries[-1]['page_end'] = page_num

                            toc_entries.append({
                                'title': title,
                                'page_start': page_num + 1,
                                'page_end': page_num + 1
                            })
                            break

            # Set final page_end
            if toc_entries:
                toc_entries[-1]['page_end'] = total_pages

        return toc_entries


# ============================================================================
# UTILITIES
# ============================================================================

def generate_analysis_plan(toc_data: dict, prompt_file: Path, config_file: Path) -> str:
    """
    Generate an analysis plan using OpenAI based on TOC and prompt.

    Args:
        toc_data: Dictionary with TOC and filing metadata
        prompt_file: Path to the prompt file
        config_file: Path to OpenAI config YAML file

    Returns:
        Analysis plan as JSON string from OpenAI

    Raises:
        ImportError: If required libraries are not installed
        FileNotFoundError: If files don't exist
        Exception: For API errors
    """
    # Check dependencies
    if not HAS_OPENAI:
        raise ImportError("OpenAI library not found. Install with: pip install openai")
    if not HAS_YAML:
        raise ImportError("PyYAML library not found. Install with: pip install pyyaml")

    # Load OpenAI config
    if not config_file.exists():
        raise FileNotFoundError(f"OpenAI config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    api_key = config.get('openai_api_key')
    model = config.get('model', 'gpt-4o-mini')
    temperature = config.get('temperature', 1.0)
    max_tokens = config.get('max_completion_tokens', 4096)

    if not api_key:
        raise ValueError("OpenAI API key not found in config file")

    # Load prompt
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, 'r') as f:
        system_prompt = f.read()

    # Prepare user message with TOC JSON (compact format to save tokens)
    toc_json = json.dumps(toc_data)
    user_message = f"Table of Contents:\n\n{toc_json}\n\nGenerate the analysis plan as specified."

    # Call OpenAI API
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens
    )

    # Extract and return the plan
    plan = response.choices[0].message.content.strip()
    return plan


def _extract_section_content(pdf_or_reader, toc_entry: dict, use_pdfplumber: bool) -> str:
    """
    Extract text content from a section using page range.

    Args:
        pdf_or_reader: PDF object (pdfplumber) or PdfReader (pypdf)
        toc_entry: TOC entry with page_start and page_end
        use_pdfplumber: True if using pdfplumber, False if using pypdf

    Returns:
        Extracted text content
    """
    page_start = toc_entry.get('page_start')
    page_end = toc_entry.get('page_end')

    if page_start is None or page_end is None:
        return ""

    extracted_text = []
    if use_pdfplumber:
        for page_num in range(page_start - 1, min(page_end, len(pdf_or_reader.pages))):
            if 0 <= page_num < len(pdf_or_reader.pages):
                page = pdf_or_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
    else:  # pypdf
        for page_num in range(page_start - 1, min(page_end, len(pdf_or_reader.pages))):
            if 0 <= page_num < len(pdf_or_reader.pages):
                page = pdf_or_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    extracted_text.append(text)

    return "\n\n".join(extracted_text)


def _analyze_batch(client, sections_data: list, analysis_prompt: str,
                  model: str, temperature: float, max_tokens: int) -> str:
    """
    Analyze a batch of sections using OpenAI.

    Args:
        client: OpenAI client
        sections_data: List of {"title": str, "content": str}
        analysis_prompt: System prompt for analysis
        model: OpenAI model name
        temperature: Temperature setting
        max_tokens: Max tokens for response

    Returns:
        Analysis text from OpenAI
    """
    # Use compact JSON (no indent) to save API tokens
    batch_json = json.dumps({"sections": sections_data})
    user_message = f"Analyze the following section(s) from the SEC Filing:\n\n{batch_json}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()


def _generate_final_report(client, all_analyses: str, final_prompt: str,
                          model: str, temperature: float, max_tokens: int) -> str:
    """
    Generate final synthesis report from all batch analyses.

    Args:
        client: OpenAI client
        all_analyses: Combined text of all batch analyses
        final_prompt: System prompt for synthesis
        model: OpenAI model name
        temperature: Temperature setting
        max_tokens: Max tokens for response

    Returns:
        Final report text from OpenAI
    """
    final_user_message = f"\nPlease synthesize these batch analyses into a comprehensive final report\n: {all_analyses}\n"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": final_user_message}
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()


def _save_report_to_pdf(report_text: str, output_path: Path) -> None:
    """
    Save markdown report to PDF file using reportlab.

    Args:
        report_text: Markdown formatted report
        output_path: Path where PDF should be saved
    """
    # Ensure output path has .pdf extension
    if output_path.suffix.lower() != '.pdf':
        output_path = output_path.with_suffix('.pdf')

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        import re

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )

        # Define styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor='#2c3e50',
            spaceAfter=12,
            alignment=TA_LEFT
        )

        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=16,
            textColor='#34495e',
            spaceAfter=10,
            spaceBefore=12
        )

        heading3_style = ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontSize=14,
            textColor='#7f8c8d',
            spaceAfter=8,
            spaceBefore=10
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=14,
            spaceAfter=6
        )

        # Parse markdown and build story
        story = []
        lines = report_text.split('\n')

        for line in lines:
            line = line.strip()

            if not line:
                story.append(Spacer(1, 0.1*inch))
                continue

            # H1 headers
            if line.startswith('# '):
                text = line[2:].strip()
                story.append(Paragraph(text, title_style))

            # H2 headers
            elif line.startswith('## '):
                text = line[3:].strip()
                story.append(Paragraph(text, heading2_style))

            # H3 headers
            elif line.startswith('### '):
                text = line[4:].strip()
                story.append(Paragraph(text, heading3_style))

            # Bullet points
            elif line.startswith('- ') or line.startswith('* '):
                text = '• ' + line[2:].strip()
                story.append(Paragraph(text, body_style))

            # Bold text (**text**)
            elif '**' in line:
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                story.append(Paragraph(text, body_style))

            # Regular text
            else:
                story.append(Paragraph(line, body_style))

        # Build PDF
        doc.build(story)
        print(f"\n✓ Report saved to PDF: {output_path}")

    except ImportError as e:
        print(f"\n✗ Warning: Could not save PDF. Missing library: {e}")
        print("Install with: pip install reportlab")

        # Fallback: save as text file
        text_path = output_path.with_suffix('.txt')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(text_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Report saved to text file: {text_path}")

    except Exception as e:
        print(f"\n✗ Error saving PDF: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: save as text file
        text_path = output_path.with_suffix('.txt')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(text_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Report saved to text file: {text_path}")


def _process_single_batch(plan_item: dict, pdf_or_reader, toc_lookup: dict,
                         use_pdfplumber: bool, client, analysis_prompt: str,
                         model: str, temperature: float, max_tokens: int, temp_file) -> None:
    """
    Process a single priority batch: extract content and analyze with AI.

    Args:
        plan_item: Dictionary with priority and sections
        pdf_or_reader: PDF reader object (pdfplumber or pypdf)
        toc_lookup: Dictionary mapping section titles to TOC entries
        use_pdfplumber: Whether using pdfplumber (vs pypdf)
        client: OpenAI client
        analysis_prompt: System prompt for batch analysis
        model: OpenAI model name
        temperature: Model temperature
        max_tokens: Max tokens for response
        temp_file: File handle to write analysis results
    """
    priority = plan_item.get('priority')
    sections = plan_item.get('sections', [])

    # Normalize sections to always be a list
    if isinstance(sections, str):
        sections = [sections]

    # Extract content for this batch
    batch_sections = []

    for section_item in sections:
        # Handle both string and dict formats from OpenAI
        if isinstance(section_item, dict):
            section_title = section_item.get('title', '')
        else:
            section_title = section_item

        # Find matching TOC entry (with fuzzy matching)
        toc_entry = toc_lookup.get(section_title)
        if not toc_entry:
            for toc_title, entry in toc_lookup.items():
                if section_title in toc_title or toc_title in section_title:
                    toc_entry = entry
                    break

        if not toc_entry or toc_entry.get('page_start') is None:
            continue

        # Extract section content
        content = _extract_section_content(pdf_or_reader, toc_entry, use_pdfplumber)
        if content:
            batch_sections.append({
                "title": section_title,
                "content": content
            })

    # Analyze batch if we have sections
    if batch_sections:
        print(f"Analyzing Priority {priority} batch ({len(batch_sections)} sections)...")
        analysis = _analyze_batch(client, batch_sections, analysis_prompt,
                                model, temperature, max_tokens)

        # Write to temp file
        temp_file.write(f"\n{'='*80}\n")
        temp_file.write(f"PRIORITY {priority} ANALYSIS\n")
        temp_file.write(f"{'='*80}\n\n")
        temp_file.write(analysis)
        temp_file.write(f"\n\n")


def _generate_and_save_final_report(client, temp_path: str, final_prompt: str,
                                    model: str, temperature: float, max_tokens: int,
                                    output_pdf: Path = None) -> None:
    """
    Generate final synthesis report from all batch analyses and save to PDF.

    Args:
        client: OpenAI client
        temp_path: Path to temp file with batch analyses
        final_prompt: System prompt for final synthesis
        model: OpenAI model name
        temperature: Model temperature
        max_tokens: Max tokens for response
        output_pdf: Optional path to save PDF report
    """
    # Read all batch analyses from temp file
    print("\nAll batches analyzed. Generating final comprehensive report...")
    with open(temp_path, 'r') as f:
        all_analyses = f.read()

    # Generate final synthesis report
    final_report = _generate_final_report(client, all_analyses, final_prompt,
                                         model, temperature, max_tokens)

    # Print the final report
    print("\n" + "="*80)
    print("COMPREHENSIVE FINAL ANALYSIS REPORT")
    print("="*80 + "\n")
    print(final_report)
    print("\n" + "="*80 + "\n")

    # Save report to PDF if output path provided
    if output_pdf:
        _save_report_to_pdf(final_report, output_pdf)

    # Clean up temp file
    import os
    os.unlink(temp_path)
    print(f"Temp file cleaned up: {temp_path}")


def analyze_sec_filing(pdf_path: Path, toc_data: dict, plan: str,
                      analysis_prompt_file: Path, final_prompt_file: Path,
                      openai_config_file: Path, output_pdf: Path = None) -> None:
    """
    Analyze SEC filing and generate comprehensive AI-powered report.

    Main workflow orchestrator that:
    1. Loads configuration and prompts
    2. Opens the PDF and builds TOC lookup
    3. Processes each priority batch (extract content + AI analysis)
    4. Generates final synthesis report with AI
    5. Saves professional PDF report

    Args:
        pdf_path: Path to the PDF file
        toc_data: Dictionary with TOC and filing metadata
        plan: Analysis plan JSON string from OpenAI
        analysis_prompt_file: Path to the batch analysis prompt file
        final_prompt_file: Path to the final synthesis prompt file
        openai_config_file: Path to OpenAI config file
        output_pdf: Optional path to save final report PDF
    """
    import tempfile

    # Check dependencies
    if not HAS_OPENAI:
        raise ImportError("OpenAI library not found. Install with: pip install openai")
    if not HAS_YAML:
        raise ImportError("PyYAML library not found. Install with: pip install pyyaml")

    # Load OpenAI config
    with open(openai_config_file, 'r') as f:
        config = yaml.safe_load(f)

    api_key = config.get('openai_api_key')
    model = config.get('model', 'gpt-5o-mini')
    temperature = config.get('temperature', 1.0)
    max_tokens = config.get('max_completion_tokens', 25600)

    # Load analysis prompts
    with open(analysis_prompt_file, 'r') as f:
        analysis_prompt = f.read()

    with open(final_prompt_file, 'r') as f:
        final_prompt = f.read()

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Create temp file to store all batch analyses
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
    temp_path = temp_file.name

    # Parse the plan JSON
    try:
        plan_items = json.loads(plan)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid plan JSON: {e}")

    # Build a lookup dictionary for TOC entries by title
    toc_lookup = {}
    for entry in toc_data.get('table_of_contents', []):
        title = entry.get('title', '')
        if title and entry.get('page_start') is not None:
            toc_lookup[title] = {
                'title': title,
                'page_start': entry.get('page_start'),
                'page_end': entry.get('page_end'),
                'level': entry.get('level')
            }

    # Open PDF based on available library
    use_pdfplumber = HAS_PDFPLUMBER
    pdf_or_reader = None

    if HAS_PDFPLUMBER:
        pdf_or_reader = pdfplumber.open(pdf_path)
    elif HAS_PYPDF:
        pdf_or_reader = PdfReader(str(pdf_path))
        use_pdfplumber = False
    else:
        raise ImportError("No PDF library available (need pdfplumber or pypdf)")

    try:
        # Process each priority batch
        for plan_item in sorted(plan_items, key=lambda x: x.get('priority', 999)):
            _process_single_batch(plan_item, pdf_or_reader, toc_lookup, use_pdfplumber,
                                client, analysis_prompt, model, temperature, max_tokens, temp_file)

    finally:
        # Close PDF/reader
        if use_pdfplumber and pdf_or_reader:
            pdf_or_reader.close()

    # Close temp file before reading
    temp_file.close()

    # Generate final report and save to PDF
    _generate_and_save_final_report(client, temp_path, final_prompt, model,
                                   temperature, max_tokens, output_pdf)


def print_structured_data_json(data: dict) -> None:
    """
    Print structured data as formatted JSON only (no headers or decorations).

    Args:
        data: Dictionary with 'filing' and 'table_of_contents' keys
    """
    print(json.dumps(data, indent=2))


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze SEC filing PDFs with AI-powered insights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze SEC filing (generates summarized_<filename>.pdf)
  python sec_filings_analyzer.py --pdf ./output/TSLA_10-Q_20251022.pdf

  # Analyze with custom output location
  python sec_filings_analyzer.py --pdf ./output/TSLA_10-Q_20251022.pdf \\
      --output ./reports/tesla_q3_analysis.pdf

  # Use custom config and prompt files
  python sec_filings_analyzer.py --pdf /path/to/10k.pdf \\
      --openai-config ./config/openai.yaml \\
      --plan-prompt ./prompts/sec_fillings_analysis_plan.txt

Requirements:
  pip install pdfplumber pypdf openai pyyaml reportlab
        '''
    )

    parser.add_argument(
        '--pdf',
        required=True,
        type=Path,
        help='Path to SEC filing PDF (10-K, 10-Q, etc.)'
    )

    parser.add_argument(
        '--openai-config',
        type=Path,
        default=Path('config/openai.yaml'),
        help='Path to OpenAI config YAML file (default: config/openai.yaml)'
    )

    parser.add_argument(
        '--plan-prompt',
        type=Path,
        default=Path('prompts/sec_fillings_analysis_plan.txt'),
        help='Path to analysis plan prompt file (default: prompts/sec_fillings_analysis_plan.txt)'
    )

    parser.add_argument(
        '--analysis-prompt',
        type=Path,
        default=Path('prompts/sec_filing_analysis.txt'),
        help='Path to content analysis prompt file (default: prompts/sec_filing_analysis.txt)'
    )

    parser.add_argument(
        '--final-prompt',
        type=Path,
        default=Path('prompts/sec_filing_analysis_final.txt'),
        help='Path to final synthesis prompt file (default: prompts/sec_filing_analysis_final.txt)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Path to save final analysis report as PDF (e.g., output/report.pdf). If not specified, defaults to "summarized_<input_pdf_name>.pdf" in the same directory as input PDF.'
    )

    args = parser.parse_args()

    # Check dependencies
    if not (HAS_PDFPLUMBER or HAS_PYPDF):
        print("ERROR: No PDF library found. Run: pip install pdfplumber pypdf", file=sys.stderr)
        return 1

    # Validate input
    if not args.pdf.exists():
        print(f"ERROR: PDF file not found: {args.pdf}", file=sys.stderr)
        return 1

    try:
        # Extract structured data (metadata + TOC)
        extractor = TOCExtractor(args.pdf)
        data = extractor.extract_structured_data()

        # Validate dependencies
        if not HAS_OPENAI:
            print("ERROR: OpenAI library not found. Install with: pip install openai", file=sys.stderr)
            return 1
        if not HAS_YAML:
            print("ERROR: PyYAML library not found. Install with: pip install pyyaml", file=sys.stderr)
            return 1

        # Generate analysis plan
        print("Generating analysis plan...")
        plan = generate_analysis_plan(data, args.plan_prompt, args.openai_config)

        # Generate output PDF path if not specified
        if args.output:
            output_pdf = args.output
        else:
            # Use input PDF name with "summarized_" prefix
            output_pdf = args.pdf.parent / f"summarized_{args.pdf.name}"
            output_pdf = output_pdf.with_suffix('.pdf')

        print(f"Output will be saved to: {output_pdf}")

        # Analyze SEC filing and generate comprehensive report
        analyze_sec_filing(args.pdf, data, plan, args.analysis_prompt,
                          args.final_prompt, args.openai_config, output_pdf)

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
